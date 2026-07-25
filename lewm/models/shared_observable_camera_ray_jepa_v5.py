"""Single-encoder JEPA and observable camera-ray evidence scaffold.

This module is intentionally additive.  It reuses the proven token-to-BEV JEPA
components and the V4 evidence contract, but the evidence head owns no image
encoder.  One online ``VisionEncoder`` call therefore supplies both consumers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import time
from typing import Any, Mapping, Sequence
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    CAMERA_HORIZONTAL_FOV_DEG,
    CAMERA_IMAGE_SHAPE,
    CAMERA_NEAR_M,
    CAMERA_VERTICAL_FOV_DEG,
    EVIDENCE_SCHEMA,
    GROUND_SUPPORT_COUNT,
    OUTPUT_CELL_SIZE_M,
    OUTPUT_FORWARD_MIN_EDGE_M,
    OUTPUT_LEFT_MIN_EDGE_M,
    OUTPUT_SHAPE,
    PIXEL_RAY_STRIDE_PX,
    RASTER_SCHEMA,
    SOURCE_CELL_SIZE_M,
    SOURCE_FORWARD_MIN_EDGE_M,
    SOURCE_LEFT_MIN_EDGE_M,
    SOURCE_SHAPE,
)
from lewm.models.egomotion_bev_jepa import EgomotionBevJepa, GLOBAL_CROSS_ATTENTION_LIFT
from lewm.models.encoders import VisionEncoder
from lewm.models.observable_camera_ray_evidence_v4 import (
    DEFAULT_QUERY_CHUNK_SIZE,
    DENSE_FEATURE_DIM,
    DEPTH_BIN_COUNT,
    DEPTH_BIN_SIZE_M,
    DEPTH_FAR_EDGE_M,
    DEPTH_NEAR_EDGE_M,
    ENCODER_DIM,
    ENCODER_HEADS,
    GROUND_HIDDEN_DIM,
    IMAGE_SIZE,
    PATCH_SIZE,
    PIXEL_RAY_SHAPE,
    TOKEN_SIDE,
    GroundQueryGeometryV4,
    ObservableCameraRayEvidenceV4Model,
    ObservableCameraRayEvidenceV4RawOutput,
)
from lewm.models.observable_camera_ray_evidence_v4_training import (
    DEFAULT_PIXEL_RAY_CHUNK_SIZE,
    HierarchicalRasterCrossEntropyV4,
    ObservableCameraRayEvidenceV4Targets,
    balanced_ground_clear_bce_v4,
    derive_observable_camera_ray_evidence_v4_targets,
    hierarchical_raster_cross_entropy_v4,
    ordered_obstacle_first_hit_nll_breakdown_v4,
    soft_rasterize_observable_camera_ray_evidence_v4,
)


CHECKPOINT_V5_SCHEMA = "lewm_go2_shared_observable_camera_ray_jepa_checkpoint_v5"
G2_GATE_REPORT_V5_SCHEMA = "lewm_go2_shared_jepa_g2_gate_report_v5"
G3_GATE_REPORT_V5_SCHEMA = "lewm_go2_shared_jepa_g3_gate_report_v5"
G2_GATE_METRICS_V5 = (
    "aggregate_physical_gate_pass_fraction",
    "per_family_physical_gate_pass_fraction",
    "jepa_health_gate_pass_fraction",
    "counterfactual_gate_pass_fraction",
)
G3_GATE_METRICS_V5 = (
    "exact_morphology_equivalence_pass_fraction",
    "configuration_runtime_gate_pass_fraction",
    "safety_gate_pass_fraction",
    "task_gate_pass_fraction",
)
PRODUCTION_MODEL_CONFIG_V5_SCHEMA = (
    "lewm_go2_shared_observable_camera_ray_jepa_production_config_v1"
)
SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA = (
    "lewm_go2_shared_observable_camera_ray_jepa_synthetic_only_config_v1"
)
MODEL_FAMILY = "shared_observable_camera_ray_jepa_v5"
LIFECYCLE_G3_CANDIDATE = "g3_candidate"
LIFECYCLE_PROMOTED = "promoted"
_SHA256_LENGTH = 64
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
_DEPLOYMENT_STATE_PREFIXES = ("encoder.", "bev_decoder.", "evidence_head.")
_TRAINING_ONLY_STATE_PREFIXES = (
    "target_encoder.",
    "target_bev_decoder.",
    "predictor.",
    "occupancy_head.",
)
_DETERMINISTIC_DEPLOYMENT_BUFFER_KEYS = (
    "bev_decoder.coordinate_features",
    "evidence_head.canonical_ground_support_xy_body_m",
)
_GATE_REPORT_MAPPING_SOURCE = "canonical_mapping"
_GATE_REPORT_FILE_SOURCE = "canonical_json_file"
_GATE_REPORT_REGISTRY_SOURCE = "filesystem_registry"
PRODUCTION_AUTHORITY_MANIFEST_V5_SCHEMA = (
    "lewm_go2_shared_jepa_repository_authority_v6"
)
DATASET_ROLE_MANIFEST_V5_SCHEMA = "lewm_go2_shared_jepa_dataset_roles_v5"
ROLE_COMMITMENT_V5_SCHEMA = "lewm_go2_shared_jepa_role_commitment_v5"
EVALUATION_PROTOCOL_V5_SCHEMA = "lewm_go2_shared_jepa_evaluation_protocol_v5"
ACCESS_LEDGER_V5_SCHEMA = "lewm_go2_shared_jepa_access_ledger_v5"
TRAINING_RUN_V5_SCHEMA = "lewm_go2_shared_jepa_training_run_v5"
RAW_GATE_RESULT_V5_SCHEMA = "lewm_go2_shared_jepa_raw_gate_result_v5"
_FORBIDDEN_ROLE_NAMES = ("heldout", "sealed")


@dataclass(frozen=True)
class V4HeadMigrationReceiptV5:
    """Proof that compatible fit-model encoder/head weights were migrated."""

    fit_model_state_sha256: str
    shared_encoder_state_sha256: str
    evidence_head_state_sha256: str
    migrated_head_key_count: int
    source_shape: tuple[int, int]
    pixel_ray_shape: tuple[int, int]


@dataclass(frozen=True)
class SharedObservableCameraRayJepaV5Config:
    """Complete constructor and preprocessing contract for one V5 model."""

    schema: str = PRODUCTION_MODEL_CONFIG_V5_SCHEMA
    image_size: int = IMAGE_SIZE
    patch_size: int = PATCH_SIZE
    encoder_dim: int = ENCODER_DIM
    encoder_depth: int = 6
    encoder_heads: int = ENCODER_HEADS
    encoder_mlp_ratio: int = 4
    bev_dim: int = 64
    bev_size: tuple[int, int] = OUTPUT_SHAPE
    forward_range_m: tuple[float, float] = (-0.95, 5.35)
    left_range_m: tuple[float, float] = (-3.15, 3.15)
    action_dim: int = 9
    bev_attention_heads: int = 4
    bev_lift_type: str = GLOBAL_CROSS_ATTENTION_LIFT
    projective_horizontal_fov_deg: float | None = None
    projective_vertical_fov_deg: float | None = None
    projective_camera_xyz_body_m: tuple[float, float, float] | None = None
    projective_camera_rpy_body_rad: tuple[float, float, float] | None = None
    projective_near_m: float | None = None
    projective_vertical_anchor_z_body_m: tuple[float, ...] | None = None
    projective_output_cell_size_m: float | None = None
    projective_footprint_radius_m: float | None = None
    projective_footprint_perimeter_samples: int | None = None
    projective_attention_sigma_tokens: float = 1.0
    projective_attention_bias_floor: float = -6.0
    predictor_hidden_dim: int = 128
    target_ema_momentum: float = 0.996
    jepa_weight: float = 1.0
    occupancy_weight: float = 0.0
    equivariance_weight: float = 0.25
    action_contrast_weight: float = 1.0
    action_margin_fraction: float = 0.1
    variance_weight: float = 0.1
    variance_target_std: float = 0.5
    source_shape: tuple[int, int] = SOURCE_SHAPE
    pixel_ray_shape: tuple[int, int] = PIXEL_RAY_SHAPE
    query_chunk_size: int | None = DEFAULT_QUERY_CHUNK_SIZE
    v4_pixel_ray_chunk_size: int = DEFAULT_PIXEL_RAY_CHUNK_SIZE
    observable_camera_ray_v4_weight: float = 1.0
    normalization_mean: tuple[float, float, float] = NORMALIZATION_MEAN
    normalization_std: tuple[float, float, float] = NORMALIZATION_STD

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema not in {
            PRODUCTION_MODEL_CONFIG_V5_SCHEMA,
            SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA,
        }:
            raise ValueError("V5 model-config schema changed")
        if self.schema == PRODUCTION_MODEL_CONFIG_V5_SCHEMA:
            changed_inputs = [
                field.name
                for field in fields(self)
                if field.name != "schema"
                and (
                    type(getattr(self, field.name)) is not type(field.default)
                    or getattr(self, field.name) != field.default
                )
            ]
            if changed_inputs:
                raise PermissionError(
                    "production V5 config changed frozen defaults at input: "
                    + ", ".join(changed_inputs)
                )
        integer_fields = (
            "image_size",
            "patch_size",
            "encoder_dim",
            "encoder_heads",
            "encoder_mlp_ratio",
            "bev_dim",
            "action_dim",
            "bev_attention_heads",
            "predictor_hidden_dim",
            "v4_pixel_ray_chunk_size",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.encoder_depth, bool)
            or not isinstance(self.encoder_depth, int)
            or self.encoder_depth < 0
        ):
            raise ValueError("encoder_depth must be a non-negative integer")
        if (
            self.image_size != IMAGE_SIZE
            or self.patch_size != PATCH_SIZE
            or self.encoder_dim != ENCODER_DIM
            or self.encoder_heads != ENCODER_HEADS
        ):
            raise ValueError("V5 must use the canonical V4 encoder geometry")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.bev_dim % self.bev_attention_heads != 0:
            raise ValueError("bev_dim must be divisible by bev_attention_heads")
        object.__setattr__(self, "bev_size", _positive_shape(self.bev_size, name="bev_size"))
        object.__setattr__(
            self, "source_shape", _positive_shape(self.source_shape, name="source_shape")
        )
        object.__setattr__(
            self,
            "pixel_ray_shape",
            _positive_shape(self.pixel_ray_shape, name="pixel_ray_shape"),
        )
        for name in ("forward_range_m", "left_range_m"):
            values = _finite_float_tuple(getattr(self, name), length=2, name=name)
            if values[1] <= values[0]:
                raise ValueError(f"{name} must be strictly increasing")
            object.__setattr__(self, name, values)
        if self.bev_lift_type != GLOBAL_CROSS_ATTENTION_LIFT:
            raise ValueError("V5 currently freezes the reviewed global BEV lift")
        projective_values = (
            self.projective_horizontal_fov_deg,
            self.projective_vertical_fov_deg,
            self.projective_camera_xyz_body_m,
            self.projective_camera_rpy_body_rad,
            self.projective_near_m,
            self.projective_vertical_anchor_z_body_m,
            self.projective_output_cell_size_m,
            self.projective_footprint_radius_m,
            self.projective_footprint_perimeter_samples,
        )
        if any(value is not None for value in projective_values):
            raise ValueError("global BEV lift must not carry projective parameters")
        if self.projective_attention_sigma_tokens != 1.0:
            raise ValueError("global BEV lift requires attention sigma 1.0")
        if self.projective_attention_bias_floor != -6.0:
            raise ValueError("global BEV lift requires attention bias floor -6.0")
        momentum = _finite_float(self.target_ema_momentum, "target_ema_momentum")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("target_ema_momentum must lie in [0,1)")
        object.__setattr__(self, "target_ema_momentum", momentum)
        positive_weights = (
            "jepa_weight",
            "equivariance_weight",
            "action_contrast_weight",
            "action_margin_fraction",
            "variance_weight",
            "variance_target_std",
            "observable_camera_ray_v4_weight",
        )
        for name in positive_weights:
            value = _finite_float(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if _finite_float(self.occupancy_weight, "occupancy_weight") != 0.0:
            raise ValueError("V5 V4 supervision requires occupancy_weight=0")
        object.__setattr__(self, "occupancy_weight", 0.0)
        if self.query_chunk_size is not None and (
            isinstance(self.query_chunk_size, bool)
            or not isinstance(self.query_chunk_size, int)
            or self.query_chunk_size <= 0
        ):
            raise ValueError("query_chunk_size must be positive or None")
        object.__setattr__(
            self,
            "normalization_mean",
            _finite_float_tuple(
                self.normalization_mean, length=3, name="normalization_mean"
            ),
        )
        std = _finite_float_tuple(
            self.normalization_std, length=3, name="normalization_std"
        )
        if any(value <= 0.0 for value in std):
            raise ValueError("normalization_std entries must be positive")
        object.__setattr__(self, "normalization_std", std)
        if self.schema == PRODUCTION_MODEL_CONFIG_V5_SCHEMA:
            changed = [
                field.name
                for field in fields(self)
                if field.name != "schema"
                and (
                    type(getattr(self, field.name)) is not type(field.default)
                    or getattr(self, field.name) != field.default
                )
            ]
            if changed:
                raise PermissionError(
                    "production V5 config changed frozen defaults: "
                    + ", ".join(changed)
                )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "bev_size",
            "forward_range_m",
            "left_range_m",
            "source_shape",
            "pixel_ray_shape",
            "normalization_mean",
            "normalization_std",
        ):
            result[name] = list(result[name])
        return result

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "SharedObservableCameraRayJepaV5Config":
        expected = set(cls().to_dict())
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("checkpoint-v5 model-config fields changed")
        payload = dict(value)
        tuple_fields = (
            "bev_size",
            "forward_range_m",
            "left_range_m",
            "source_shape",
            "pixel_ray_shape",
            "normalization_mean",
            "normalization_std",
        )
        for name in tuple_fields:
            raw = payload[name]
            if not isinstance(raw, list):
                raise ValueError(f"model config {name} must be a list")
            payload[name] = tuple(raw)
        return cls(**payload)


@dataclass(frozen=True)
class SharedOnlineFrameV5:
    patch_tokens: torch.Tensor
    bev: torch.Tensor
    evidence: ObservableCameraRayEvidenceV4RawOutput
    camera_origin_body_m: torch.Tensor
    camera_basis_body_fru: torch.Tensor
    ground_plane_z_body_m: torch.Tensor


@dataclass(frozen=True)
class JepaCounterfactualsV5:
    wrong_action_contrast_loss: torch.Tensor | None
    zero_action_contrast_loss: torch.Tensor | None
    wrong_action_advantage_over_target_change: torch.Tensor | None
    wrong_commanded_delta_advantage_over_target_change: torch.Tensor | None
    wrong_action_prediction_sensitivity: torch.Tensor | None
    wrong_commanded_delta_prediction_sensitivity: torch.Tensor | None


@dataclass(frozen=True)
class EstablishedJepaPackageV5:
    total: torch.Tensor
    prediction: torch.Tensor
    equivariance: torch.Tensor
    action_contrast: torch.Tensor
    variance: torch.Tensor
    warped_persistence: torch.Tensor
    prediction_to_persistence_ratio: torch.Tensor
    prediction_valid_cells: torch.Tensor
    target_cross_sample_std_mean: torch.Tensor
    target_cross_sample_effective_rank: torch.Tensor
    counterfactuals: JepaCounterfactualsV5


@dataclass(frozen=True)
class SharedTrainingPairV5:
    current: SharedOnlineFrameV5
    next: SharedOnlineFrameV5
    predicted_next_bev: torch.Tensor
    stop_gradient_target_next_bev: torch.Tensor
    commanded_warped_current_bev: torch.Tensor
    commanded_overlap_mask: torch.Tensor
    realized_warped_current_bev: torch.Tensor
    realized_overlap_mask: torch.Tensor
    jepa: EstablishedJepaPackageV5

    @property
    def jepa_loss(self) -> torch.Tensor:
        return self.jepa.prediction


@dataclass(frozen=True)
class SharedHierarchicalV4LossV5:
    """Derived-raster diagnostic only; it is not a complete V4 training loss."""

    current: HierarchicalRasterCrossEntropyV4
    next: HierarchicalRasterCrossEntropyV4
    total: torch.Tensor


@dataclass(frozen=True)
class ObservableCameraRayV4FrameSupervisionV5:
    """Raw observable labels plus the derived-raster label for one frame."""

    pixel_hit_mask: torch.Tensor
    pixel_first_hit_distance_m: torch.Tensor
    ground_support_in_frustum: torch.Tensor
    ground_support_clear_to_target: torch.Tensor
    target_raster_labels: torch.Tensor


@dataclass(frozen=True)
class SharedObservableCameraRayV4FrameLossV5:
    """The frozen four-equal V4 objective for one shared-encoder frame."""

    ordered_first_hit_nll: torch.Tensor
    target_bin_offset_smooth_l1: torch.Tensor
    ground_clear_distance_state_balanced_bce: torch.Tensor
    derived_raster_hierarchical_bce: HierarchicalRasterCrossEntropyV4
    total: torch.Tensor


@dataclass(frozen=True)
class SharedObservableCameraRayV4LossV5:
    """Complete current/next V4 supervision used by the joint objective."""

    current: SharedObservableCameraRayV4FrameLossV5
    next: SharedObservableCameraRayV4FrameLossV5
    total: torch.Tensor


@dataclass(frozen=True)
class SharedJointLossV5:
    total: torch.Tensor
    established_jepa: EstablishedJepaPackageV5
    observable_camera_ray_v4: SharedObservableCameraRayV4LossV5
    observable_camera_ray_v4_weight: float


def _skew_balanced_pixel_offset_loss_v5(
    predicted_offset_m: torch.Tensor,
    targets: ObservableCameraRayEvidenceV4Targets,
) -> torch.Tensor:
    """Match the V4 fit objective's equal weighting over represented bins."""

    if (
        not isinstance(predicted_offset_m, torch.Tensor)
        or predicted_offset_m.ndim != 4
        or not predicted_offset_m.is_floating_point()
    ):
        raise ValueError("predicted offsets must have shape (B,D,H,W)")
    expected = (
        predicted_offset_m.shape[0],
        predicted_offset_m.shape[2],
        predicted_offset_m.shape[3],
    )
    if tuple(targets.pixel_in_range_hit_mask.shape) != expected:
        raise ValueError("pixel targets do not match predicted offsets")
    if predicted_offset_m.device != targets.pixel_hit_bin_index.device:
        raise ValueError("pixel predictions and targets must share a device")
    selected = predicted_offset_m.gather(
        1,
        targets.pixel_hit_bin_index[:, None],
    ).squeeze(1)
    group_losses = []
    for depth_bin in range(predicted_offset_m.shape[1]):
        mask = targets.pixel_in_range_hit_mask & (
            targets.pixel_hit_bin_index == depth_bin
        )
        if bool(mask.any().item()):
            group_losses.append(
                F.smooth_l1_loss(
                    selected[mask],
                    targets.pixel_within_bin_offset_m[mask].to(
                        dtype=selected.dtype
                    ),
                    beta=0.01,
                    reduction="mean",
                )
            )
    return (
        torch.stack(group_losses).mean()
        if group_losses
        else predicted_offset_m.sum() * 0.0
    )


@dataclass(frozen=True)
class CheckpointProvenanceV5:
    """Content commitments required by every checkpoint-v5 lifecycle."""

    dataset_manifest_sha256: str
    corpus_plan_sha256: str
    geometry_contract_sha256: str
    camera_calibration_sha256: str
    implementation_sha256: str
    fit_gate_report_sha256: str
    v4_fit_checkpoint_sha256: str
    training_run_sha256: str
    gate_attempt_registry_source_sha256: str
    g2_role_commitment_sha256: str
    g3_role_commitment_sha256: str
    g2_evaluation_protocol_sha256: str
    g3_evaluation_protocol_sha256: str
    g2_finalizer_source_sha256: str
    g3_finalizer_source_sha256: str
    training_scene_ids: tuple[str, ...]
    schema: str = "lewm_go2_shared_jepa_checkpoint_provenance_v1"

    def __post_init__(self) -> None:
        if self.schema != "lewm_go2_shared_jepa_checkpoint_provenance_v1":
            raise ValueError("checkpoint-v5 provenance schema changed")
        for name in (
            "dataset_manifest_sha256",
            "corpus_plan_sha256",
            "geometry_contract_sha256",
            "camera_calibration_sha256",
            "implementation_sha256",
            "fit_gate_report_sha256",
            "v4_fit_checkpoint_sha256",
            "training_run_sha256",
            "gate_attempt_registry_source_sha256",
            "g2_role_commitment_sha256",
            "g3_role_commitment_sha256",
            "g2_evaluation_protocol_sha256",
            "g3_evaluation_protocol_sha256",
            "g2_finalizer_source_sha256",
            "g3_finalizer_source_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if (
            not self.training_scene_ids
            or tuple(sorted(self.training_scene_ids)) != self.training_scene_ids
            or len(set(self.training_scene_ids)) != len(self.training_scene_ids)
            or any(not isinstance(item, str) or not item for item in self.training_scene_ids)
        ):
            raise ValueError("training_scene_ids must be nonempty, sorted, and unique")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["training_scene_ids"] = list(self.training_scene_ids)
        return result

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


class ObservableCameraRayEvidenceV4Head(nn.Module):
    """Encoder-free V4 head consuming ordered patch-7 image tokens."""

    _validate_calibration = staticmethod(
        ObservableCameraRayEvidenceV4Model._validate_calibration
    )
    ground_query_geometry = ObservableCameraRayEvidenceV4Model.ground_query_geometry
    pixel_branch = ObservableCameraRayEvidenceV4Model.pixel_branch
    ground_branch = ObservableCameraRayEvidenceV4Model.ground_branch

    def __init__(
        self,
        *,
        source_shape: tuple[int, int] = SOURCE_SHAPE,
        pixel_ray_shape: tuple[int, int] = PIXEL_RAY_SHAPE,
        query_chunk_size: int | None = DEFAULT_QUERY_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        self.source_shape = _positive_shape(source_shape, name="source_shape")
        self.pixel_ray_shape = _positive_shape(
            pixel_ray_shape, name="pixel_ray_shape"
        )
        self.query_chunk_size = (
            None if query_chunk_size is None else int(query_chunk_size)
        )
        if self.query_chunk_size is not None and self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive or None")

        self.dense_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                ENCODER_DIM,
                DENSE_FEATURE_DIM,
                kernel_size=PATCH_SIZE,
                stride=PATCH_SIZE,
            ),
            nn.GroupNorm(4, DENSE_FEATURE_DIM),
            nn.GELU(),
            nn.Conv2d(
                DENSE_FEATURE_DIM,
                DENSE_FEATURE_DIM,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(4, DENSE_FEATURE_DIM),
            nn.GELU(),
        )
        self.pixel_head = nn.Conv2d(
            DENSE_FEATURE_DIM,
            2 * DEPTH_BIN_COUNT,
            kernel_size=1,
        )
        self.ground_head = nn.Sequential(
            nn.Linear(DENSE_FEATURE_DIM + 4, GROUND_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(GROUND_HIDDEN_DIM, 1),
        )
        self._init_heads()
        support = ObservableCameraRayEvidenceV4Model._canonical_support_xy(
            self.source_shape
        )
        self.register_buffer(
            "canonical_ground_support_xy_body_m", support, persistent=True
        )

    def _init_heads(self) -> None:
        nn.init.xavier_uniform_(self.pixel_head.weight)
        nn.init.zeros_(self.pixel_head.bias)
        with torch.no_grad():
            self.pixel_head.bias[:DEPTH_BIN_COUNT] = -math.log(
                float(DEPTH_BIN_COUNT - 1)
            )
        for module in self.ground_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def decode_dense_features(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(patch_tokens, torch.Tensor)
            or patch_tokens.ndim != 3
            or tuple(patch_tokens.shape[1:])
            != (TOKEN_SIDE * TOKEN_SIDE, ENCODER_DIM)
            or not patch_tokens.is_floating_point()
        ):
            raise ValueError(
                "patch_tokens must have shape (B,256,192) and floating dtype"
            )
        token_map = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0], ENCODER_DIM, TOKEN_SIDE, TOKEN_SIDE
        )
        dense = self.dense_decoder(token_map)
        expected = (patch_tokens.shape[0], DENSE_FEATURE_DIM, IMAGE_SIZE, IMAGE_SIZE)
        if tuple(dense.shape) != expected:
            raise RuntimeError("V4 dense decoder returned an unexpected shape")
        return dense

    def forward(
        self,
        patch_tokens: torch.Tensor,
        camera_origin_body_m: torch.Tensor,
        camera_basis_body_fru: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
        *,
        query_chunk_size: int | None = None,
    ) -> ObservableCameraRayEvidenceV4RawOutput:
        dense = self.decode_dense_features(patch_tokens)
        query = self.ground_query_geometry(
            camera_origin_body_m,
            camera_basis_body_fru,
            ground_plane_z_body_m,
        )
        if query.in_frustum.shape[0] != patch_tokens.shape[0]:
            raise ValueError("patch-token and calibration batches differ")
        hazard, offset = self.pixel_branch(dense)
        ground = self.ground_branch(
            dense,
            query,
            query_chunk_size=query_chunk_size,
        )
        return ObservableCameraRayEvidenceV4RawOutput(
            pixel_first_hit_hazard_logits=hazard,
            pixel_within_bin_offset_m=offset,
            ground_clear_to_target_logits=ground,
            ground_query_in_frustum=query.in_frustum,
            ground_query_uv_px=query.uv_px,
            ground_target_distance_m=query.target_distance_m,
        )

    def migrate_from_fit_model(
        self,
        fit_model: ObservableCameraRayEvidenceV4Model,
    ) -> tuple[str, ...]:
        """Copy every compatible non-encoder state entry, rejecting ambiguity."""

        if not isinstance(fit_model, ObservableCameraRayEvidenceV4Model):
            raise TypeError("fit_model must be an ObservableCameraRayEvidenceV4Model")
        if (
            tuple(fit_model.source_shape) != self.source_shape
            or tuple(fit_model.pixel_ray_shape) != self.pixel_ray_shape
        ):
            raise ValueError("fit-model and encoder-free head geometries differ")
        source = {
            name: value
            for name, value in fit_model.state_dict().items()
            if not name.startswith("encoder.")
        }
        expected = set(self.state_dict())
        if set(source) != expected:
            raise ValueError("fit-model non-encoder state contract changed")
        self.load_state_dict(source, strict=True)
        return tuple(sorted(source))


class SharedObservableCameraRayJepaV5(EgomotionBevJepa):
    """Established EMA JEPA with one online encoder shared by the V4 head."""

    def __init__(
        self,
        config: SharedObservableCameraRayJepaV5Config | None = None,
    ) -> None:
        config = config or SharedObservableCameraRayJepaV5Config()
        if not isinstance(config, SharedObservableCameraRayJepaV5Config):
            raise TypeError("config must be SharedObservableCameraRayJepaV5Config")
        super().__init__(
            image_size=config.image_size,
            patch_size=config.patch_size,
            encoder_dim=config.encoder_dim,
            encoder_depth=config.encoder_depth,
            encoder_heads=config.encoder_heads,
            encoder_mlp_ratio=config.encoder_mlp_ratio,
            bev_dim=config.bev_dim,
            bev_size=config.bev_size,
            forward_range_m=config.forward_range_m,
            left_range_m=config.left_range_m,
            action_dim=config.action_dim,
            bev_attention_heads=config.bev_attention_heads,
            bev_lift_type=config.bev_lift_type,
            projective_horizontal_fov_deg=config.projective_horizontal_fov_deg,
            projective_vertical_fov_deg=config.projective_vertical_fov_deg,
            projective_camera_xyz_body_m=config.projective_camera_xyz_body_m,
            projective_camera_rpy_body_rad=config.projective_camera_rpy_body_rad,
            projective_near_m=config.projective_near_m,
            projective_vertical_anchor_z_body_m=(
                config.projective_vertical_anchor_z_body_m
            ),
            projective_output_cell_size_m=config.projective_output_cell_size_m,
            projective_footprint_radius_m=config.projective_footprint_radius_m,
            projective_footprint_perimeter_samples=(
                config.projective_footprint_perimeter_samples
            ),
            projective_attention_sigma_tokens=(
                config.projective_attention_sigma_tokens
            ),
            projective_attention_bias_floor=(
                config.projective_attention_bias_floor
            ),
            predictor_hidden_dim=config.predictor_hidden_dim,
            target_ema_momentum=config.target_ema_momentum,
            jepa_weight=config.jepa_weight,
            occupancy_weight=config.occupancy_weight,
            equivariance_weight=config.equivariance_weight,
            action_contrast_weight=config.action_contrast_weight,
            action_margin_fraction=config.action_margin_fraction,
            variance_weight=config.variance_weight,
            variance_target_std=config.variance_target_std,
        )
        self.model_config = config
        self.evidence_head = ObservableCameraRayEvidenceV4Head(
            source_shape=config.source_shape,
            pixel_ray_shape=config.pixel_ray_shape,
            query_chunk_size=config.query_chunk_size,
        )
        self._capture_online = False
        self._captured_online: tuple[torch.Tensor, torch.Tensor] | None = None
        self._require_encoder_contract()

    def _require_encoder_contract(self) -> None:
        encoders = [
            name
            for name, module in self.named_modules()
            if isinstance(module, VisionEncoder)
        ]
        if encoders != ["encoder", "target_encoder"]:
            raise RuntimeError(
                "V5 requires one online and one training-only target encoder: "
                f"{encoders}"
            )
        if any(
            isinstance(module, VisionEncoder)
            for module in self.evidence_head.modules()
        ):
            raise RuntimeError("V4 evidence head must not own a visual encoder")
        for module in (self.target_encoder, self.target_bev_decoder):
            if module.training or any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                raise RuntimeError("EMA target modules must stay frozen and in eval mode")

    def train(self, mode: bool = True) -> "SharedObservableCameraRayJepaV5":
        super().train(mode)
        self._require_encoder_contract()
        return self

    def _encode_online(
        self,
        image: torch.Tensor,
        base_quat_world_xyzw: torch.Tensor | None = None,
        stored_base_yaw_rad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ObservableCameraRayEvidenceV4Model._validate_image(image)
        tokens = self.encoder.forward_tokens(image)[:, 1:]
        bev = self.bev_decoder(
            tokens,
            base_quat_world_xyzw=base_quat_world_xyzw,
            stored_base_yaw_rad=stored_base_yaw_rad,
        )
        if self._capture_online:
            if self._captured_online is not None:
                raise RuntimeError("established JEPA made more than one online encode call")
            self._captured_online = (tokens, bev)
        return bev

    def _frame_from_online(
        self,
        tokens: torch.Tensor,
        bev: torch.Tensor,
        camera_origin_body_m: torch.Tensor,
        camera_basis_body_fru: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
    ) -> SharedOnlineFrameV5:
        evidence = self.evidence_head(
            tokens,
            camera_origin_body_m,
            camera_basis_body_fru,
            ground_plane_z_body_m,
        )
        return SharedOnlineFrameV5(
            patch_tokens=tokens,
            bev=bev,
            evidence=evidence,
            camera_origin_body_m=camera_origin_body_m,
            camera_basis_body_fru=camera_basis_body_fru,
            ground_plane_z_body_m=ground_plane_z_body_m,
        )

    def forward_frame(
        self,
        image: torch.Tensor,
        camera_origin_body_m: torch.Tensor,
        camera_basis_body_fru: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
    ) -> SharedOnlineFrameV5:
        ObservableCameraRayEvidenceV4Model._validate_image(image)
        tokens = self.encoder.forward_tokens(image)[:, 1:]
        bev = self.bev_decoder(tokens)
        return self._frame_from_online(
            tokens,
            bev,
            camera_origin_body_m,
            camera_basis_body_fru,
            ground_plane_z_body_m,
        )

    def forward_training_pair(
        self,
        current_image: torch.Tensor,
        next_image: torch.Tensor,
        action: torch.Tensor,
        realized_delta_pose_current: torch.Tensor,
        *,
        commanded_delta_pose_current: torch.Tensor,
        current_camera_origin_body_m: torch.Tensor,
        current_camera_basis_body_fru: torch.Tensor,
        current_ground_plane_z_body_m: torch.Tensor,
        next_camera_origin_body_m: torch.Tensor,
        next_camera_basis_body_fru: torch.Tensor,
        next_ground_plane_z_body_m: torch.Tensor,
        next_prediction_mask: torch.Tensor | None = None,
        diagnostic_wrong_action: torch.Tensor | None = None,
        diagnostic_wrong_action_delta_pose_current: torch.Tensor | None = None,
        diagnostic_wrong_commanded_delta_pose_current: torch.Tensor | None = None,
    ) -> SharedTrainingPairV5:
        self._require_encoder_contract()
        if (
            diagnostic_wrong_action is None
            or diagnostic_wrong_action_delta_pose_current is None
            or diagnostic_wrong_commanded_delta_pose_current is None
        ):
            raise ValueError(
                "V5 training requires wrong-action and wrong-commanded-delta "
                "counterfactuals"
            )
        if self._capture_online or self._captured_online is not None:
            raise RuntimeError("online token capture is not reentrant")
        self._capture_online = True
        try:
            established = super().forward(
                current_image,
                next_image,
                action,
                realized_delta_pose_current,
                commanded_delta_pose_current=commanded_delta_pose_current,
                next_prediction_mask=next_prediction_mask,
                diagnostic_wrong_action=diagnostic_wrong_action,
                diagnostic_wrong_action_delta_pose_current=(
                    diagnostic_wrong_action_delta_pose_current
                ),
                diagnostic_wrong_commanded_delta_pose_current=(
                    diagnostic_wrong_commanded_delta_pose_current
                ),
            )
            captured = self._captured_online
        finally:
            self._capture_online = False
            self._captured_online = None
        if captured is None:
            raise RuntimeError("established JEPA did not expose its online token pass")
        tokens, online_bev = captured
        batch = current_image.shape[0]
        if tokens.shape[0] != 2 * batch or online_bev.shape[0] != 2 * batch:
            raise RuntimeError("established JEPA online batch contract changed")
        combined_evidence = self.evidence_head(
            tokens,
            torch.cat(
                (current_camera_origin_body_m, next_camera_origin_body_m), dim=0
            ),
            torch.cat(
                (current_camera_basis_body_fru, next_camera_basis_body_fru), dim=0
            ),
            torch.cat(
                (current_ground_plane_z_body_m, next_ground_plane_z_body_m), dim=0
            ),
        )
        current = SharedOnlineFrameV5(
            patch_tokens=tokens[:batch],
            bev=online_bev[:batch],
            evidence=_slice_evidence(combined_evidence, slice(0, batch)),
            camera_origin_body_m=current_camera_origin_body_m,
            camera_basis_body_fru=current_camera_basis_body_fru,
            ground_plane_z_body_m=current_ground_plane_z_body_m,
        )
        next_frame = SharedOnlineFrameV5(
            patch_tokens=tokens[batch:],
            bev=online_bev[batch:],
            evidence=_slice_evidence(combined_evidence, slice(batch, 2 * batch)),
            camera_origin_body_m=next_camera_origin_body_m,
            camera_basis_body_fru=next_camera_basis_body_fru,
            ground_plane_z_body_m=next_ground_plane_z_body_m,
        )
        target_std, target_rank = _established_target_health(
            established["target_next_bev"]
        )
        counterfactuals = JepaCounterfactualsV5(
            wrong_action_contrast_loss=established.get("wrong_action_contrast_loss"),
            zero_action_contrast_loss=established.get("zero_action_contrast_loss"),
            wrong_action_advantage_over_target_change=established.get(
                "wrong_action_advantage_over_target_change"
            ),
            wrong_commanded_delta_advantage_over_target_change=established.get(
                "wrong_delta_advantage_over_target_change"
            ),
            wrong_action_prediction_sensitivity=established.get(
                "wrong_action_prediction_sensitivity"
            ),
            wrong_commanded_delta_prediction_sensitivity=established.get(
                "wrong_delta_prediction_sensitivity"
            ),
        )
        package = EstablishedJepaPackageV5(
            total=established["loss"],
            prediction=established["jepa_loss"],
            equivariance=established["equivariance_loss"],
            action_contrast=established["action_contrast_loss"],
            variance=established["variance_loss"],
            warped_persistence=established["warped_persistence_loss"],
            prediction_to_persistence_ratio=established[
                "prediction_to_persistence_ratio"
            ],
            prediction_valid_cells=established["prediction_valid_cells"],
            target_cross_sample_std_mean=target_std,
            target_cross_sample_effective_rank=target_rank,
            counterfactuals=counterfactuals,
        )
        return SharedTrainingPairV5(
            current=current,
            next=next_frame,
            predicted_next_bev=established["predicted_next_bev"],
            stop_gradient_target_next_bev=established["target_next_bev"],
            commanded_warped_current_bev=established[
                "commanded_warped_current_bev"
            ],
            commanded_overlap_mask=established["prediction_valid_mask"],
            realized_warped_current_bev=established["realized_warped_current_bev"],
            realized_overlap_mask=established[
                "realized_equivariance_valid_mask"
            ],
            jepa=package,
        )

    def hierarchical_v4_loss(
        self,
        pair: SharedTrainingPairV5,
        current_target_labels: torch.Tensor,
        next_target_labels: torch.Tensor,
    ) -> SharedHierarchicalV4LossV5:
        if not isinstance(pair, SharedTrainingPairV5):
            raise TypeError("pair must be SharedTrainingPairV5")
        losses = []
        for frame, labels in (
            (pair.current, current_target_labels),
            (pair.next, next_target_labels),
        ):
            raster = soft_rasterize_observable_camera_ray_evidence_v4(
                frame.evidence,
                camera_origin_body_m=frame.camera_origin_body_m,
                camera_basis_body_fru=frame.camera_basis_body_fru,
                pixel_ray_chunk_size=self.model_config.v4_pixel_ray_chunk_size,
            )
            losses.append(hierarchical_raster_cross_entropy_v4(raster, labels))
        total = 0.5 * (losses[0].total + losses[1].total)
        return SharedHierarchicalV4LossV5(
            current=losses[0],
            next=losses[1],
            total=total,
        )

    def observable_camera_ray_v4_loss(
        self,
        pair: SharedTrainingPairV5,
        current_supervision: ObservableCameraRayV4FrameSupervisionV5,
        next_supervision: ObservableCameraRayV4FrameSupervisionV5,
    ) -> SharedObservableCameraRayV4LossV5:
        """Reproduce the complete four-equal V4 objective on both frames."""

        if not isinstance(pair, SharedTrainingPairV5):
            raise TypeError("pair must be SharedTrainingPairV5")
        frame_losses = []
        for frame, supervision in (
            (pair.current, current_supervision),
            (pair.next, next_supervision),
        ):
            if not isinstance(
                supervision, ObservableCameraRayV4FrameSupervisionV5
            ):
                raise TypeError(
                    "V4 supervision must be ObservableCameraRayV4FrameSupervisionV5"
                )
            targets = derive_observable_camera_ray_evidence_v4_targets(
                pixel_hit_mask=supervision.pixel_hit_mask,
                pixel_first_hit_distance_m=supervision.pixel_first_hit_distance_m,
                ground_support_in_frustum=supervision.ground_support_in_frustum,
                ground_support_clear_to_target=(
                    supervision.ground_support_clear_to_target
                ),
            )
            evidence = frame.evidence
            if not torch.equal(
                evidence.ground_query_in_frustum,
                targets.ground_in_frustum,
            ):
                raise ValueError(
                    "model calibration does not reproduce V4 ground visibility"
                )
            ordered = ordered_obstacle_first_hit_nll_breakdown_v4(
                evidence.pixel_first_hit_hazard_logits,
                targets,
            ).total
            offset = _skew_balanced_pixel_offset_loss_v5(
                evidence.pixel_within_bin_offset_m,
                targets,
            )
            ground = balanced_ground_clear_bce_v4(
                evidence.ground_clear_to_target_logits,
                targets,
                evidence.ground_target_distance_m,
            )
            raster = soft_rasterize_observable_camera_ray_evidence_v4(
                evidence,
                camera_origin_body_m=frame.camera_origin_body_m,
                camera_basis_body_fru=frame.camera_basis_body_fru,
                pixel_ray_chunk_size=self.model_config.v4_pixel_ray_chunk_size,
            )
            derived = hierarchical_raster_cross_entropy_v4(
                raster,
                supervision.target_raster_labels,
            )
            total = 0.25 * (ordered + offset + ground + derived.total)
            frame_losses.append(
                SharedObservableCameraRayV4FrameLossV5(
                    ordered_first_hit_nll=ordered,
                    target_bin_offset_smooth_l1=offset,
                    ground_clear_distance_state_balanced_bce=ground,
                    derived_raster_hierarchical_bce=derived,
                    total=total,
                )
            )
        total = 0.5 * (frame_losses[0].total + frame_losses[1].total)
        if not bool(torch.isfinite(total).item()):
            raise ValueError("complete V4 loss must be finite")
        return SharedObservableCameraRayV4LossV5(
            current=frame_losses[0],
            next=frame_losses[1],
            total=total,
        )

    def combine_joint_losses(
        self,
        pair: SharedTrainingPairV5,
        current_v4_supervision: ObservableCameraRayV4FrameSupervisionV5,
        next_v4_supervision: ObservableCameraRayV4FrameSupervisionV5,
    ) -> SharedJointLossV5:
        if not isinstance(pair, SharedTrainingPairV5):
            raise TypeError("pair must be SharedTrainingPairV5")
        observable_camera_ray_v4_loss = self.observable_camera_ray_v4_loss(
            pair,
            current_v4_supervision,
            next_v4_supervision,
        )
        weight = self.model_config.observable_camera_ray_v4_weight
        return SharedJointLossV5(
            total=pair.jepa.total + weight * observable_camera_ray_v4_loss.total,
            established_jepa=pair.jepa,
            observable_camera_ray_v4=observable_camera_ray_v4_loss,
            observable_camera_ray_v4_weight=weight,
        )

    @torch.no_grad()
    def hard_sync_ema_target_from_online(self) -> None:
        self.target_encoder.load_state_dict(self.encoder.state_dict(), strict=True)
        self.target_bev_decoder.load_state_dict(
            self.bev_decoder.state_dict(), strict=True
        )
        for module in (self.target_encoder, self.target_bev_decoder):
            module.requires_grad_(False)
            module.eval()

    @torch.no_grad()
    def update_ema_target_after_optimizer_step(self) -> None:
        """The sole supported post-optimizer EMA update entry point."""

        super().update_target_encoder()
        for module in (self.target_encoder, self.target_bev_decoder):
            module.requires_grad_(False)
            module.eval()
        self._require_encoder_contract()

    def update_target_encoder(self) -> None:
        raise RuntimeError(
            "use update_ema_target_after_optimizer_step immediately after optimizer.step"
        )

    def deployment_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        deployment = {
            name: value.detach().cpu().clone()
            for name, value in state.items()
            if name.startswith(_DEPLOYMENT_STATE_PREFIXES)
        }
        if not deployment or any(
            name.startswith(_TRAINING_ONLY_STATE_PREFIXES) for name in deployment
        ):
            raise RuntimeError("deployment state filtering failed")
        deployment_buffers = {
            name
            for name, _value in self.named_buffers(remove_duplicate=False)
            if name in deployment
        }
        if deployment_buffers != set(_DETERMINISTIC_DEPLOYMENT_BUFFER_KEYS):
            raise RuntimeError("deployment deterministic-buffer inventory changed")
        canonical_buffers = _canonical_deterministic_deployment_buffers(
            self.model_config
        )
        for name in _DETERMINISTIC_DEPLOYMENT_BUFFER_KEYS:
            if not torch.equal(deployment[name], canonical_buffers[name]):
                raise ValueError(
                    f"deployment deterministic buffer {name} changed from canonical"
                )
        return deployment

    def load_deployment_state_dict(
        self, state: Mapping[str, torch.Tensor]
    ) -> None:
        expected = self.deployment_state_dict()
        _validate_state_contract(state, expected)
        for prefix, module in (
            ("encoder.", self.encoder),
            ("bev_decoder.", self.bev_decoder),
            ("evidence_head.", self.evidence_head),
        ):
            local = {
                name[len(prefix) :]: value
                for name, value in state.items()
                if name.startswith(prefix)
            }
            module.load_state_dict(local, strict=True)

    def migrate_from_fit_model(
        self,
        fit_model: ObservableCameraRayEvidenceV4Model,
    ) -> V4HeadMigrationReceiptV5:
        """Migrate the fit encoder and all compatible V4 head state exactly."""

        if not isinstance(fit_model, ObservableCameraRayEvidenceV4Model):
            raise TypeError("fit_model must be an ObservableCameraRayEvidenceV4Model")
        self.encoder.load_state_dict(fit_model.encoder.state_dict(), strict=True)
        migrated = self.evidence_head.migrate_from_fit_model(fit_model)
        self.hard_sync_ema_target_from_online()
        return V4HeadMigrationReceiptV5(
            fit_model_state_sha256=tensor_state_dict_sha256(fit_model.state_dict()),
            shared_encoder_state_sha256=tensor_state_dict_sha256(
                self.encoder.state_dict()
            ),
            evidence_head_state_sha256=tensor_state_dict_sha256(
                self.evidence_head.state_dict()
            ),
            migrated_head_key_count=len(migrated),
            source_shape=self.evidence_head.source_shape,
            pixel_ray_shape=self.evidence_head.pixel_ray_shape,
        )


def tensor_state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Content hash over ordered tensor names, dtypes, shapes, and CPU bytes."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError("state dict must be a nonempty mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise ValueError("state dict entries must be named tensors")
        tensor = value.detach().cpu().contiguous()
        header = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        encoded = json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        )
    return digest.hexdigest()


def canonical_v4_geometry_contract_v5(
    config: SharedObservableCameraRayJepaV5Config,
) -> dict[str, Any]:
    return {
        "schema": "lewm_go2_observable_camera_ray_v4_geometry_v1",
        "evidence_schema": EVIDENCE_SCHEMA,
        "raster_schema": RASTER_SCHEMA,
        "image_size": config.image_size,
        "normalization_mean": list(config.normalization_mean),
        "normalization_std": list(config.normalization_std),
        "source_shape": list(config.source_shape),
        "source_cell_size_m": SOURCE_CELL_SIZE_M,
        "source_forward_min_edge_m": SOURCE_FORWARD_MIN_EDGE_M,
        "source_left_min_edge_m": SOURCE_LEFT_MIN_EDGE_M,
        "ground_support_count": GROUND_SUPPORT_COUNT,
        "pixel_ray_shape": list(config.pixel_ray_shape),
        "camera_image_shape": list(CAMERA_IMAGE_SHAPE),
        "pixel_ray_stride_px": PIXEL_RAY_STRIDE_PX,
        "camera_horizontal_fov_deg": CAMERA_HORIZONTAL_FOV_DEG,
        "camera_vertical_fov_deg": CAMERA_VERTICAL_FOV_DEG,
        "camera_near_m": CAMERA_NEAR_M,
        "depth_bin_count": DEPTH_BIN_COUNT,
        "depth_bin_size_m": DEPTH_BIN_SIZE_M,
        "depth_near_edge_m": DEPTH_NEAR_EDGE_M,
        "depth_far_edge_m": DEPTH_FAR_EDGE_M,
        "output_shape": list(OUTPUT_SHAPE),
        "output_cell_size_m": OUTPUT_CELL_SIZE_M,
        "output_forward_min_edge_m": OUTPUT_FORWARD_MIN_EDGE_M,
        "output_left_min_edge_m": OUTPUT_LEFT_MIN_EDGE_M,
        "physical_target_inflation_m": 0.0,
    }


def canonical_bev_geometry_contract_v5(
    config: SharedObservableCameraRayJepaV5Config,
) -> dict[str, Any]:
    return {
        "schema": "lewm_go2_shared_jepa_bev_geometry_v1",
        "bev_size": list(config.bev_size),
        "forward_center_range_m": list(config.forward_range_m),
        "left_center_range_m": list(config.left_range_m),
        "lift_type": config.bev_lift_type,
        "attention_heads": config.bev_attention_heads,
        "projective_horizontal_fov_deg": config.projective_horizontal_fov_deg,
        "projective_vertical_fov_deg": config.projective_vertical_fov_deg,
        "projective_camera_xyz_body_m": config.projective_camera_xyz_body_m,
        "projective_camera_rpy_body_rad": config.projective_camera_rpy_body_rad,
        "projective_near_m": config.projective_near_m,
        "projective_vertical_anchor_z_body_m": (
            config.projective_vertical_anchor_z_body_m
        ),
        "projective_output_cell_size_m": config.projective_output_cell_size_m,
        "projective_footprint_radius_m": config.projective_footprint_radius_m,
        "projective_footprint_perimeter_samples": (
            config.projective_footprint_perimeter_samples
        ),
        "projective_attention_sigma_tokens": (
            config.projective_attention_sigma_tokens
        ),
        "projective_attention_bias_floor": config.projective_attention_bias_floor,
    }


def shared_output_contract_v5(
    model: SharedObservableCameraRayJepaV5,
) -> dict[str, Any]:
    config = model.model_config
    return {
        "schema": "lewm_go2_shared_observable_camera_ray_jepa_output_v2",
        "online_frame_fields": [
            "patch_tokens",
            "bev",
            "evidence",
            "camera_origin_body_m",
            "camera_basis_body_fru",
            "ground_plane_z_body_m",
        ],
        "patch_token_shape": ["B", TOKEN_SIDE * TOKEN_SIDE, ENCODER_DIM],
        "bev_shape": ["B", config.bev_dim, *config.bev_size],
        "evidence_schema": EVIDENCE_SCHEMA,
        "evidence_fields": {
            "pixel_first_hit_hazard_logits": [
                "B",
                DEPTH_BIN_COUNT,
                *config.pixel_ray_shape,
            ],
            "pixel_within_bin_offset_m": [
                "B",
                DEPTH_BIN_COUNT,
                *config.pixel_ray_shape,
            ],
            "ground_clear_to_target_logits": [
                "B",
                *config.source_shape,
                GROUND_SUPPORT_COUNT,
            ],
            "ground_query_in_frustum": [
                "B",
                *config.source_shape,
                GROUND_SUPPORT_COUNT,
            ],
            "ground_query_uv_px": [
                "B",
                *config.source_shape,
                GROUND_SUPPORT_COUNT,
                2,
            ],
            "ground_target_distance_m": [
                "B",
                *config.source_shape,
                GROUND_SUPPORT_COUNT,
            ],
        },
        "training_pair_fields": [
            "current",
            "next",
            "predicted_next_bev",
            "stop_gradient_target_next_bev",
            "commanded_warped_current_bev",
            "commanded_overlap_mask",
            "realized_warped_current_bev",
            "realized_overlap_mask",
            "jepa",
        ],
        "established_jepa_losses": [
            "prediction",
            "equivariance",
            "action_contrast",
            "variance",
        ],
        "established_jepa_health": [
            "warped_persistence",
            "prediction_to_persistence_ratio",
            "target_cross_sample_std_mean",
            "target_cross_sample_effective_rank",
        ],
        "counterfactuals": [
            "wrong_action",
            "zero_action",
            "wrong_commanded_delta",
        ],
        "v4_supervision": {
            "objective": "four_equal_observable_camera_ray_v4",
            "required_components": [
                "ordered_first_hit_nll",
                "target_bin_offset_smooth_l1",
                "ground_clear_distance_state_balanced_bce",
                "derived_raster_hierarchical_bce",
            ],
            "frame_aggregation": "equal_current_next",
        },
        "target_output": "training_only_no_grad_ema_next_bev",
    }


def shared_architecture_contract_v5(
    model: SharedObservableCameraRayJepaV5,
) -> dict[str, Any]:
    if not isinstance(model, SharedObservableCameraRayJepaV5):
        raise TypeError("model must be SharedObservableCameraRayJepaV5")
    model._require_encoder_contract()
    config = model.model_config
    output = shared_output_contract_v5(model)
    v4_geometry = canonical_v4_geometry_contract_v5(config)
    bev_geometry = canonical_bev_geometry_contract_v5(config)
    return {
        "schema": "lewm_go2_shared_observable_camera_ray_jepa_architecture_v1",
        "model_family": MODEL_FAMILY,
        "established_jepa_base": "lewm.models.egomotion_bev_jepa.EgomotionBevJepa",
        "online_vision_encoder_count": 1,
        "training_target_vision_encoder_count": 1,
        "v4_head_vision_encoder_count": 0,
        "online_encoder_call_contract": "one_current_next_batched_online_call_v1",
        "target_encoder_call_contract": "one_next_only_no_grad_target_call_v1",
        "jepa_target_mode": "training_only_ema_encoder_and_bev_decoder_v1",
        "target_ema_momentum": config.target_ema_momentum,
        "target_modules_always_eval_and_frozen": True,
        "deployment_state_prefixes": list(_DEPLOYMENT_STATE_PREFIXES),
        "training_only_state_prefixes": list(_TRAINING_ONLY_STATE_PREFIXES),
        "deterministic_deployment_buffer_keys": list(
            _DETERMINISTIC_DEPLOYMENT_BUFFER_KEYS
        ),
        "model_config_sha256": config.content_sha256,
        "output_contract_sha256": _canonical_sha256(output),
        "v4_geometry_sha256": _canonical_sha256(v4_geometry),
        "bev_geometry_sha256": _canonical_sha256(bev_geometry),
    }


def checkpoint_contract_bindings_v5(
    model: SharedObservableCameraRayJepaV5,
    provenance: CheckpointProvenanceV5,
) -> dict[str, str]:
    if not isinstance(provenance, CheckpointProvenanceV5):
        raise TypeError("provenance must be CheckpointProvenanceV5")
    state = model.deployment_state_dict()
    return {
        "model_state_sha256": tensor_state_dict_sha256(state),
        "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
        "model_config_sha256": model.model_config.content_sha256,
        "architecture_contract_sha256": _canonical_sha256(
            shared_architecture_contract_v5(model)
        ),
        "output_contract_sha256": _canonical_sha256(
            shared_output_contract_v5(model)
        ),
        "provenance_contract_sha256": provenance.content_sha256,
        "implementation_sha256": provenance.implementation_sha256,
        "v4_geometry_sha256": _canonical_sha256(
            canonical_v4_geometry_contract_v5(model.model_config)
        ),
        "bev_geometry_sha256": _canonical_sha256(
            canonical_bev_geometry_contract_v5(model.model_config)
        ),
    }


def _gate_report_schema_v5(gate: str) -> str:
    if gate == "g2":
        return G2_GATE_REPORT_V5_SCHEMA
    if gate == "g3":
        return G3_GATE_REPORT_V5_SCHEMA
    raise ValueError("gate report must name g2 or g3")


def gate_thresholds_v5(gate: str) -> dict[str, Any]:
    names = G2_GATE_METRICS_V5 if gate == "g2" else G3_GATE_METRICS_V5 if gate == "g3" else None
    if names is None:
        raise ValueError("gate thresholds must name g2 or g3")
    return {
        "schema": f"lewm_go2_shared_jepa_{gate}_thresholds_v5",
        "metrics": {
            name: {"comparison": "greater_than_or_equal", "value": 1.0}
            for name in names
        },
    }


def _gate_authority_bindings_v5(
    provenance: CheckpointProvenanceV5,
    *,
    gate: str,
) -> dict[str, str]:
    return {
        "attempt_registry_source_sha256": (
            provenance.gate_attempt_registry_source_sha256
        ),
        "gate_role_commitment_sha256": getattr(
            provenance, f"{gate}_role_commitment_sha256"
        ),
        "evaluation_protocol_sha256": getattr(
            provenance, f"{gate}_evaluation_protocol_sha256"
        ),
        "finalizer_source_sha256": getattr(
            provenance, f"{gate}_finalizer_source_sha256"
        ),
        "threshold_contract_sha256": _canonical_sha256(
            gate_thresholds_v5(gate)
        ),
        "implementation_sha256": provenance.implementation_sha256,
    }


def _validate_gate_report_v5(
    value: object,
    *,
    gate: str,
    caller_content_sha256: str,
    bindings: Mapping[str, str],
    authority_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Snapshot and validate an actual canonical external gate report."""

    _require_sha256(caller_content_sha256, name=f"{gate} caller content hash")
    expected_fields = {
        "schema",
        "gate",
        "passed",
        "model_family",
        "content_sha256",
        "metrics",
        "thresholds",
        "checks",
        "decision_sha256",
        "attempt_registry",
        "finalizer",
        *bindings,
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"{gate} gate-report fields changed")
    try:
        report = json.loads(_canonical_json_bytes(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{gate} gate report is not canonical JSON data") from exc
    if not isinstance(report, dict) or set(report) != expected_fields:
        raise ValueError(f"{gate} gate-report fields changed")
    if (
        report.get("schema") != _gate_report_schema_v5(gate)
        or report.get("gate") != gate
        or report.get("model_family") != MODEL_FAMILY
    ):
        raise ValueError(f"{gate} gate-report identity changed")
    if report.get("passed") is not True:
        raise PermissionError(f"{gate} gate report does not record a pass")
    thresholds = gate_thresholds_v5(gate)
    if report.get("thresholds") != thresholds:
        raise PermissionError(f"{gate} gate thresholds changed")
    metric_names = set(thresholds["metrics"])
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != metric_names:
        raise ValueError(f"{gate} gate metrics changed")
    normalized_metrics: dict[str, float] = {}
    for name in sorted(metric_names):
        raw_metric = metrics[name]
        if isinstance(raw_metric, bool) or not isinstance(raw_metric, (int, float)):
            raise ValueError(f"{gate} metric {name} must be numeric")
        metric = float(raw_metric)
        if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
            raise ValueError(f"{gate} metric {name} must lie in [0,1]")
        normalized_metrics[name] = metric
    checks = {
        name: normalized_metrics[name] >= float(thresholds["metrics"][name]["value"])
        for name in sorted(metric_names)
    }
    if report.get("checks") != checks or not all(checks.values()):
        raise PermissionError(f"{gate} measured metrics do not pass frozen thresholds")
    metrics_sha256 = _canonical_sha256(normalized_metrics)
    thresholds_sha256 = _canonical_sha256(thresholds)
    decision_sha256 = _canonical_sha256(
        {
            "schema": f"lewm_go2_shared_jepa_{gate}_decision_v5",
            "metrics_sha256": metrics_sha256,
            "thresholds_sha256": thresholds_sha256,
            "checks": checks,
            "passed": True,
        }
    )
    if report.get("decision_sha256") != decision_sha256:
        raise ValueError(f"{gate} gate decision hash changed")
    _validate_gate_attempt_registry_v5(
        report.get("attempt_registry"),
        gate=gate,
        bindings=bindings,
        authority_bindings=authority_bindings,
    )
    _validate_gate_finalizer_v5(
        report.get("finalizer"),
        gate=gate,
        attempt_registry=report["attempt_registry"],
        metrics_sha256=metrics_sha256,
        thresholds_sha256=thresholds_sha256,
        decision_sha256=decision_sha256,
        authority_bindings=authority_bindings,
    )
    report_content_sha256 = report.get("content_sha256")
    _require_sha256(report_content_sha256, name=f"{gate} report content hash")
    content = dict(report)
    del content["content_sha256"]
    computed_content_sha256 = _canonical_sha256(content)
    if (
        report_content_sha256 != computed_content_sha256
        or caller_content_sha256 != computed_content_sha256
    ):
        raise ValueError(f"{gate} gate-report content hash changed")
    for name, expected in bindings.items():
        _require_sha256(report.get(name), name=f"{gate} report {name}")
        if report[name] != expected:
            raise PermissionError(f"{gate} gate report is bound to another candidate")
    return report


def _validate_gate_attempt_registry_v5(
    value: object,
    *,
    gate: str,
    bindings: Mapping[str, str],
    authority_bindings: Mapping[str, str],
) -> None:
    expected_fields = {
        "schema",
        "gate",
        "status",
        "attempt_id_sha256",
        "attempt_registry_source_sha256",
        "dataset_manifest_sha256",
        "gate_role_commitment_sha256",
        "evaluation_protocol_sha256",
        "threshold_contract_sha256",
        "model_state_sha256",
        "reserved_before_payload_access",
        "prior_attempt_count",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"{gate} attempt-registry fields changed")
    if (
        value.get("schema") != f"lewm_go2_shared_jepa_{gate}_attempt_registry_v5"
        or value.get("gate") != gate
        or value.get("status") != "consumed_once_finalized"
        or value.get("reserved_before_payload_access") is not True
        or value.get("prior_attempt_count") != 0
    ):
        raise PermissionError(f"{gate} attempt registry is not a one-shot finalization")
    expected_attempt_id = _canonical_sha256(
        {
            "schema": f"lewm_go2_shared_jepa_{gate}_attempt_identity_v5",
            "dataset_manifest_sha256": bindings["dataset_manifest_sha256"],
            "gate_role_commitment_sha256": authority_bindings[
                "gate_role_commitment_sha256"
            ],
            "evaluation_protocol_sha256": authority_bindings[
                "evaluation_protocol_sha256"
            ],
        }
    )
    expected = {
        "attempt_id_sha256": expected_attempt_id,
        "attempt_registry_source_sha256": authority_bindings[
            "attempt_registry_source_sha256"
        ],
        "dataset_manifest_sha256": bindings["dataset_manifest_sha256"],
        "gate_role_commitment_sha256": authority_bindings[
            "gate_role_commitment_sha256"
        ],
        "evaluation_protocol_sha256": authority_bindings[
            "evaluation_protocol_sha256"
        ],
        "threshold_contract_sha256": authority_bindings[
            "threshold_contract_sha256"
        ],
        "model_state_sha256": bindings["model_state_sha256"],
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise PermissionError(f"{gate} attempt registry binding changed")
    content = dict(value)
    claimed = content.pop("content_sha256", None)
    if claimed != _canonical_sha256(content):
        raise ValueError(f"{gate} attempt-registry content hash changed")


def _validate_gate_finalizer_v5(
    value: object,
    *,
    gate: str,
    attempt_registry: Mapping[str, Any],
    metrics_sha256: str,
    thresholds_sha256: str,
    decision_sha256: str,
    authority_bindings: Mapping[str, str],
) -> None:
    expected_fields = {
        "schema",
        "gate",
        "status",
        "finalizer_source_sha256",
        "implementation_sha256",
        "attempt_registry_content_sha256",
        "raw_result_content_sha256",
        "metrics_sha256",
        "thresholds_sha256",
        "decision_sha256",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"{gate} finalizer fields changed")
    if (
        value.get("schema") != f"lewm_go2_shared_jepa_{gate}_finalizer_v5"
        or value.get("gate") != gate
        or value.get("status") != "independently_finalized"
    ):
        raise PermissionError(f"{gate} report was not independently finalized")
    for name, expected in {
        "finalizer_source_sha256": authority_bindings["finalizer_source_sha256"],
        "implementation_sha256": authority_bindings["implementation_sha256"],
        "attempt_registry_content_sha256": attempt_registry["content_sha256"],
        "metrics_sha256": metrics_sha256,
        "thresholds_sha256": thresholds_sha256,
        "decision_sha256": decision_sha256,
    }.items():
        if value.get(name) != expected:
            raise PermissionError(f"{gate} finalizer {name} binding changed")
    _require_sha256(
        value.get("raw_result_content_sha256"),
        name=f"{gate} finalizer raw-result hash",
    )
    content = dict(value)
    claimed = content.pop("content_sha256", None)
    if claimed != _canonical_sha256(content):
        raise ValueError(f"{gate} finalizer content hash changed")


def _resolve_gate_report_v5(
    source: Mapping[str, Any] | str | Path,
    *,
    gate: str,
    caller_content_sha256: str,
    caller_file_sha256: str | None,
    bindings: Mapping[str, str],
    authority_bindings: Mapping[str, str],
) -> tuple[dict[str, Any], str, str | None]:
    if isinstance(source, Mapping):
        if caller_file_sha256 is not None:
            raise ValueError(f"{gate} mapping report must not carry a file hash")
        return (
            _validate_gate_report_v5(
                source,
                gate=gate,
                caller_content_sha256=caller_content_sha256,
                bindings=bindings,
                authority_bindings=authority_bindings,
            ),
            _GATE_REPORT_MAPPING_SOURCE,
            None,
        )
    if not isinstance(source, (str, Path)):
        raise TypeError(f"{gate} report must be a mapping or canonical JSON file")
    if caller_file_sha256 is None:
        raise ValueError(f"{gate} file report requires a caller file hash")
    _require_sha256(caller_file_sha256, name=f"{gate} caller file hash")
    path = Path(source)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(
            f"{gate} gate-report path must be a readable regular non-symlink file"
        ) from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError(
                    f"{gate} gate-report path must be a regular non-symlink file"
                )
            encoded = handle.read()
    except OSError as exc:
        raise ValueError(f"{gate} gate-report file could not be read") from exc
    actual_file_sha256 = hashlib.sha256(encoded).hexdigest()
    if actual_file_sha256 != caller_file_sha256:
        raise ValueError(f"{gate} gate-report file hash changed")
    try:
        parsed = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{gate} gate-report file is not strict UTF-8 JSON") from exc
    report = _validate_gate_report_v5(
        parsed,
        gate=gate,
        caller_content_sha256=caller_content_sha256,
        bindings=bindings,
        authority_bindings=authority_bindings,
    )
    if encoded != _canonical_json_bytes(report) + b"\n":
        raise ValueError(f"{gate} gate-report file is not canonical JSON")
    return report, _GATE_REPORT_FILE_SOURCE, actual_file_sha256


def _validate_embedded_gate_report_file_hash_v5(
    report: Mapping[str, Any],
    *,
    gate: str,
    source: object,
    file_sha256: object,
) -> None:
    if source == _GATE_REPORT_MAPPING_SOURCE:
        if file_sha256 is not None:
            raise ValueError(f"{gate} mapping report carries a file hash")
        return
    if source != _GATE_REPORT_FILE_SOURCE:
        raise ValueError(f"{gate} embedded gate-report source changed")
    if file_sha256 is None:
        raise ValueError(f"{gate} file report lost its file hash")
    _require_sha256(file_sha256, name=f"{gate} embedded report file hash")
    expected = hashlib.sha256(_canonical_json_bytes(report) + b"\n").hexdigest()
    if file_sha256 != expected:
        raise ValueError(f"{gate} embedded gate-report file hash changed")


def _build_checkpoint_v5_payload_structure_only_for_tests(
    model: SharedObservableCameraRayJepaV5,
    *,
    lifecycle: str,
    provenance: CheckpointProvenanceV5,
    g2_report: Mapping[str, Any] | str | Path,
    g2_report_content_sha256: str,
    g2_report_file_sha256: str | None = None,
    g3_report: Mapping[str, Any] | str | Path | None = None,
    g3_report_content_sha256: str | None = None,
    g3_report_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a validated candidate or promoted payload; no filename inference."""

    if lifecycle not in {LIFECYCLE_G3_CANDIDATE, LIFECYCLE_PROMOTED}:
        raise ValueError("checkpoint lifecycle must be g3_candidate or promoted")
    if not isinstance(provenance, CheckpointProvenanceV5):
        raise TypeError("provenance must be CheckpointProvenanceV5")
    _require_production_checkpoint_config(model.model_config)
    state = model.deployment_state_dict()
    _validate_state_contract(state, model.deployment_state_dict())
    state_sha = tensor_state_dict_sha256(state)
    model_config = model.model_config.to_dict()
    output_contract = shared_output_contract_v5(model)
    v4_geometry = canonical_v4_geometry_contract_v5(model.model_config)
    bev_geometry = canonical_bev_geometry_contract_v5(model.model_config)
    architecture = shared_architecture_contract_v5(model)
    bindings = checkpoint_contract_bindings_v5(model, provenance)
    g2_authority_bindings = _gate_authority_bindings_v5(provenance, gate="g2")
    (
        normalized_g2,
        normalized_g2_source,
        normalized_g2_file_sha256,
    ) = _resolve_gate_report_v5(
        g2_report,
        gate="g2",
        caller_content_sha256=g2_report_content_sha256,
        caller_file_sha256=g2_report_file_sha256,
        bindings=bindings,
        authority_bindings=g2_authority_bindings,
    )
    runtime_ready = lifecycle == LIFECYCLE_PROMOTED
    if runtime_ready:
        if g3_report is None or g3_report_content_sha256 is None:
            raise PermissionError("promoted checkpoint requires a passing G3 report")
        g3_authority_bindings = _gate_authority_bindings_v5(
            provenance, gate="g3"
        )
        (
            normalized_g3,
            normalized_g3_source,
            normalized_g3_file_sha256,
        ) = _resolve_gate_report_v5(
            g3_report,
            gate="g3",
            caller_content_sha256=g3_report_content_sha256,
            caller_file_sha256=g3_report_file_sha256,
            bindings=bindings,
            authority_bindings=g3_authority_bindings,
        )
    elif any(
        value is not None
        for value in (
            g3_report,
            g3_report_content_sha256,
            g3_report_file_sha256,
        )
    ):
        raise ValueError("G3 candidate must not carry a promoted G3 report")
    else:
        normalized_g3 = None
        normalized_g3_source = None
        normalized_g3_file_sha256 = None
    payload = {
        "schema": CHECKPOINT_V5_SCHEMA,
        "lifecycle": lifecycle,
        "model_family": MODEL_FAMILY,
        "runtime_ready": runtime_ready,
        "model_state_sha256": state_sha,
        "model_state_dict": state,
        "model_config": model_config,
        "model_config_sha256": model.model_config.content_sha256,
        "architecture_contract": architecture,
        "architecture_contract_sha256": _canonical_sha256(architecture),
        "output_contract": output_contract,
        "output_contract_sha256": _canonical_sha256(output_contract),
        "v4_geometry": v4_geometry,
        "v4_geometry_sha256": _canonical_sha256(v4_geometry),
        "bev_geometry": bev_geometry,
        "bev_geometry_sha256": _canonical_sha256(bev_geometry),
        "provenance": provenance.to_dict(),
        "provenance_contract_sha256": provenance.content_sha256,
        "g2_report": normalized_g2,
        "g2_report_content_sha256": g2_report_content_sha256,
        "g2_report_source": normalized_g2_source,
        "g2_report_file_sha256": normalized_g2_file_sha256,
        "g3_report": normalized_g3,
        "g3_report_content_sha256": g3_report_content_sha256,
        "g3_report_source": normalized_g3_source,
        "g3_report_file_sha256": normalized_g3_file_sha256,
    }
    _validate_checkpoint_v5_payload_structure_only_for_tests(
        payload, expected_lifecycle=lifecycle
    )
    _weights_only_roundtrip_without_recursion(payload, lifecycle=lifecycle)
    return payload


def _validate_checkpoint_v5_payload_structure_only_for_tests(
    payload: Mapping[str, Any],
    *,
    expected_lifecycle: str | None = None,
) -> None:
    expected_fields = {
        "schema",
        "lifecycle",
        "model_family",
        "runtime_ready",
        "model_state_sha256",
        "model_state_dict",
        "model_config",
        "model_config_sha256",
        "architecture_contract",
        "architecture_contract_sha256",
        "output_contract",
        "output_contract_sha256",
        "v4_geometry",
        "v4_geometry_sha256",
        "bev_geometry",
        "bev_geometry_sha256",
        "provenance",
        "provenance_contract_sha256",
        "g2_report",
        "g2_report_content_sha256",
        "g2_report_source",
        "g2_report_file_sha256",
        "g3_report",
        "g3_report_content_sha256",
        "g3_report_source",
        "g3_report_file_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("checkpoint-v5 fields changed")
    lifecycle = payload.get("lifecycle")
    if expected_lifecycle is not None and lifecycle != expected_lifecycle:
        raise PermissionError("checkpoint-v5 lifecycle does not match loader mode")
    if lifecycle not in {LIFECYCLE_G3_CANDIDATE, LIFECYCLE_PROMOTED}:
        raise ValueError("checkpoint-v5 lifecycle is invalid")
    if (
        payload.get("schema") != CHECKPOINT_V5_SCHEMA
        or payload.get("model_family") != MODEL_FAMILY
    ):
        raise ValueError("checkpoint-v5 identity changed")
    config = SharedObservableCameraRayJepaV5Config.from_mapping(
        _require_mapping(payload.get("model_config"), "model_config")
    )
    if payload.get("model_config_sha256") != config.content_sha256:
        raise ValueError("checkpoint-v5 model-config hash changed")
    _require_production_checkpoint_config(config)
    model = SharedObservableCameraRayJepaV5(config)
    expected_output = shared_output_contract_v5(model)
    _require_canonical_mapping(
        payload.get("output_contract"), expected_output, name="output contract"
    )
    if payload.get("output_contract_sha256") != _canonical_sha256(expected_output):
        raise ValueError("checkpoint-v5 output-contract hash changed")
    expected_v4_geometry = canonical_v4_geometry_contract_v5(config)
    _require_canonical_mapping(
        payload.get("v4_geometry"), expected_v4_geometry, name="V4 geometry"
    )
    if payload.get("v4_geometry_sha256") != _canonical_sha256(expected_v4_geometry):
        raise ValueError("checkpoint-v5 V4 geometry hash changed")
    expected_bev_geometry = canonical_bev_geometry_contract_v5(config)
    _require_canonical_mapping(
        payload.get("bev_geometry"), expected_bev_geometry, name="BEV geometry"
    )
    if payload.get("bev_geometry_sha256") != _canonical_sha256(expected_bev_geometry):
        raise ValueError("checkpoint-v5 BEV geometry hash changed")
    expected_architecture = shared_architecture_contract_v5(model)
    _require_canonical_mapping(
        payload.get("architecture_contract"),
        expected_architecture,
        name="architecture contract",
    )
    if payload.get("architecture_contract_sha256") != _canonical_sha256(
        expected_architecture
    ):
        raise ValueError("checkpoint-v5 architecture hash changed")
    state = _require_mapping(payload.get("model_state_dict"), "model_state_dict")
    _validate_state_contract(state, model.deployment_state_dict())
    state_sha = tensor_state_dict_sha256(state)
    if payload.get("model_state_sha256") != state_sha:
        raise ValueError("checkpoint-v5 model state hash changed")
    model.load_deployment_state_dict(state)
    if tensor_state_dict_sha256(model.deployment_state_dict()) != state_sha:
        raise ValueError("checkpoint-v5 strict state load changed bytes")
    provenance = _validate_checkpoint_provenance(payload.get("provenance"))
    if payload.get("provenance_contract_sha256") != provenance.content_sha256:
        raise ValueError("checkpoint-v5 provenance hash changed")
    bindings = {
        "model_state_sha256": state_sha,
        "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
        "model_config_sha256": config.content_sha256,
        "architecture_contract_sha256": _canonical_sha256(expected_architecture),
        "output_contract_sha256": _canonical_sha256(expected_output),
        "provenance_contract_sha256": provenance.content_sha256,
        "implementation_sha256": provenance.implementation_sha256,
        "v4_geometry_sha256": _canonical_sha256(expected_v4_geometry),
        "bev_geometry_sha256": _canonical_sha256(expected_bev_geometry),
    }
    g2_content_sha256 = payload.get("g2_report_content_sha256")
    if not isinstance(g2_content_sha256, str):
        raise ValueError("checkpoint-v5 G2 report content hash is missing")
    g2 = _validate_gate_report_v5(
        payload.get("g2_report"),
        gate="g2",
        caller_content_sha256=g2_content_sha256,
        bindings=bindings,
        authority_bindings=_gate_authority_bindings_v5(provenance, gate="g2"),
    )
    _validate_embedded_gate_report_file_hash_v5(
        g2,
        gate="g2",
        source=payload.get("g2_report_source"),
        file_sha256=payload.get("g2_report_file_sha256"),
    )
    promoted = lifecycle == LIFECYCLE_PROMOTED
    if payload.get("runtime_ready") is not promoted:
        raise PermissionError("checkpoint-v5 runtime readiness contradicts lifecycle")
    raw_g3 = payload.get("g3_report")
    raw_g3_content_sha256 = payload.get("g3_report_content_sha256")
    raw_g3_source = payload.get("g3_report_source")
    raw_g3_file_sha256 = payload.get("g3_report_file_sha256")
    if promoted:
        if not isinstance(raw_g3, Mapping) or not isinstance(
            raw_g3_content_sha256, str
        ):
            raise PermissionError("promoted checkpoint lacks a G3 report")
        g3 = _validate_gate_report_v5(
            raw_g3,
            gate="g3",
            caller_content_sha256=raw_g3_content_sha256,
            bindings=bindings,
            authority_bindings=_gate_authority_bindings_v5(
                provenance, gate="g3"
            ),
        )
        _validate_embedded_gate_report_file_hash_v5(
            g3,
            gate="g3",
            source=raw_g3_source,
            file_sha256=raw_g3_file_sha256,
        )
    elif any(
        value is not None
        for value in (
            raw_g3,
            raw_g3_content_sha256,
            raw_g3_source,
            raw_g3_file_sha256,
        )
    ):
        raise PermissionError("G3 candidate contains a promoted G3 report")


def _checkpoint_v5_weights_only_roundtrip_structure_only_for_tests(
    payload: Mapping[str, Any],
    *,
    expected_lifecycle: str | None = None,
) -> dict[str, Any]:
    _validate_checkpoint_v5_payload_structure_only_for_tests(
        payload, expected_lifecycle=expected_lifecycle
    )
    return _weights_only_roundtrip_without_recursion(
        payload, lifecycle=expected_lifecycle
    )


def _slice_evidence(
    output: ObservableCameraRayEvidenceV4RawOutput,
    index: slice,
) -> ObservableCameraRayEvidenceV4RawOutput:
    return ObservableCameraRayEvidenceV4RawOutput(
        pixel_first_hit_hazard_logits=output.pixel_first_hit_hazard_logits[index],
        pixel_within_bin_offset_m=output.pixel_within_bin_offset_m[index],
        ground_clear_to_target_logits=output.ground_clear_to_target_logits[index],
        ground_query_in_frustum=output.ground_query_in_frustum[index],
        ground_query_uv_px=output.ground_query_uv_px[index],
        ground_target_distance_m=output.ground_target_distance_m[index],
    )


@torch.no_grad()
def _established_target_health(
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the established G2 target std and effective-rank controls."""

    target_float = target.float()
    if target_float.shape[0] < 2:
        zero = target_float.new_zeros(())
        return zero, zero
    target_std = target_float.std(dim=0, unbiased=False).mean()
    centered = target_float - target_float.mean(dim=0, keepdim=True)
    samples = centered.permute(0, 2, 3, 1).reshape(-1, centered.shape[1])
    if samples.shape[0] > 65_536:
        stride = math.ceil(samples.shape[0] / 65_536)
        samples = samples[::stride]
    covariance = samples.T @ samples / max(1, samples.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if not bool((total > 0).item()):
        rank = target_float.new_zeros(())
    else:
        probabilities = eigenvalues / total
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum()
        rank = torch.exp(entropy)
    return target_std.to(target.dtype), rank.to(target.dtype)


def _validate_checkpoint_provenance(value: object) -> CheckpointProvenanceV5:
    expected = {
        "dataset_manifest_sha256",
        "corpus_plan_sha256",
        "geometry_contract_sha256",
        "camera_calibration_sha256",
        "implementation_sha256",
        "fit_gate_report_sha256",
        "v4_fit_checkpoint_sha256",
        "training_run_sha256",
        "gate_attempt_registry_source_sha256",
        "g2_role_commitment_sha256",
        "g3_role_commitment_sha256",
        "g2_evaluation_protocol_sha256",
        "g3_evaluation_protocol_sha256",
        "g2_finalizer_source_sha256",
        "g3_finalizer_source_sha256",
        "training_scene_ids",
        "schema",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("checkpoint-v5 provenance fields changed")
    scenes = value.get("training_scene_ids")
    if not isinstance(scenes, list):
        raise ValueError("checkpoint-v5 training scenes must be a list")
    return CheckpointProvenanceV5(
        dataset_manifest_sha256=value["dataset_manifest_sha256"],
        corpus_plan_sha256=value["corpus_plan_sha256"],
        geometry_contract_sha256=value["geometry_contract_sha256"],
        camera_calibration_sha256=value["camera_calibration_sha256"],
        implementation_sha256=value["implementation_sha256"],
        fit_gate_report_sha256=value["fit_gate_report_sha256"],
        v4_fit_checkpoint_sha256=value["v4_fit_checkpoint_sha256"],
        training_run_sha256=value["training_run_sha256"],
        gate_attempt_registry_source_sha256=value[
            "gate_attempt_registry_source_sha256"
        ],
        g2_role_commitment_sha256=value["g2_role_commitment_sha256"],
        g3_role_commitment_sha256=value["g3_role_commitment_sha256"],
        g2_evaluation_protocol_sha256=value["g2_evaluation_protocol_sha256"],
        g3_evaluation_protocol_sha256=value["g3_evaluation_protocol_sha256"],
        g2_finalizer_source_sha256=value["g2_finalizer_source_sha256"],
        g3_finalizer_source_sha256=value["g3_finalizer_source_sha256"],
        training_scene_ids=tuple(scenes),
        schema=value["schema"],
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_float_tuple(
    value: Sequence[float],
    *,
    length: int,
    name: str,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    return tuple(_finite_float(item, name) for item in value)


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint-v5 {name} must be a mapping")
    return value


def _require_canonical_mapping(
    value: object,
    expected: Mapping[str, Any],
    *,
    name: str,
) -> None:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise ValueError(f"checkpoint-v5 {name} changed")


def _validate_state_contract(
    state: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> None:
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("checkpoint-v5 deployment state keys changed")
    for name in sorted(expected):
        value = state[name]
        reference = expected[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint-v5 state {name} is not a tensor")
        if value.layout is not torch.strided:
            raise ValueError(f"checkpoint-v5 state {name} must be strided")
        if value.shape != reference.shape:
            raise ValueError(f"checkpoint-v5 state {name} shape changed")
        if value.dtype != reference.dtype:
            raise ValueError(f"checkpoint-v5 state {name} dtype changed")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError(f"checkpoint-v5 state {name} contains non-finite values")
        if name in _DETERMINISTIC_DEPLOYMENT_BUFFER_KEYS and not torch.equal(
            value.detach().cpu(), reference.detach().cpu()
        ):
            raise ValueError(
                f"checkpoint-v5 deterministic buffer {name} changed from canonical"
            )


def _canonical_deterministic_deployment_buffers(
    config: SharedObservableCameraRayJepaV5Config,
) -> dict[str, torch.Tensor]:
    forward = torch.linspace(
        *config.forward_range_m,
        config.bev_size[0],
        dtype=torch.float32,
    )
    left = torch.linspace(
        *config.left_range_m,
        config.bev_size[1],
        dtype=torch.float32,
    )
    forward_grid, left_grid = torch.meshgrid(forward, left, indexing="ij")
    forward_grid = forward_grid / max(abs(value) for value in config.forward_range_m)
    left_grid = left_grid / max(abs(value) for value in config.left_range_m)
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
    support = ObservableCameraRayEvidenceV4Model._canonical_support_xy(
        config.source_shape
    )
    return {
        "bev_decoder.coordinate_features": coordinate_features,
        "evidence_head.canonical_ground_support_xy_body_m": support,
    }


def _require_production_checkpoint_config(
    config: SharedObservableCameraRayJepaV5Config,
) -> None:
    if config.schema != PRODUCTION_MODEL_CONFIG_V5_SCHEMA:
        raise PermissionError(
            "checkpoint-v5 candidate/promoted lifecycle requires production config"
        )
    changed = [
        field.name
        for field in fields(config)
        if field.name != "schema"
        and (
            type(getattr(config, field.name)) is not type(field.default)
            or getattr(config, field.name) != field.default
        )
    ]
    if changed:
        raise PermissionError(
            "checkpoint-v5 production defaults changed: " + ", ".join(changed)
        )


def _weights_only_roundtrip_without_recursion(
    payload: Mapping[str, Any],
    *,
    lifecycle: str | None,
) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)
    if not isinstance(restored, dict):
        raise ValueError("checkpoint-v5 weights-only roundtrip changed root type")
    _validate_checkpoint_v5_payload_structure_only_for_tests(
        restored, expected_lifecycle=lifecycle
    )
    return restored


def _positive_shape(value: Sequence[int], *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two dimensions")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item <= 0
        for item in value
    ):
        raise ValueError(f"{name} dimensions must be positive integers")
    result = tuple(int(item) for item in value)
    return result  # type: ignore[return-value]


def _require_sha256(value: object, *, name: str) -> None:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


# ---------------------------------------------------------------------------
# Production filesystem authority and independently reconstructed gate chain.
# The earlier private helpers preserve pure shape-contract tests. Only the
# public definitions below may build or load production checkpoints.


@dataclass(frozen=True)
class _RemovedCallerFilesystemCheckpointContextV5:
    """Non-instantiable tombstone for the rejected caller-root V5 candidate."""

    artifact_root: Path
    registry_root: Path
    authority_manifest_path: str
    authority_manifest_file_sha256: str

    def __post_init__(self) -> None:
        raise PermissionError(
            "caller-created filesystem authority was removed; use the canonical factory"
        )


def _canonical_relative_path_v5(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"{name} must be a canonical relative POSIX path")
    return normalized


def _open_root_directory_v5(root: Path, *, name: str) -> int:
    try:
        metadata = os.lstat(root)
    except OSError as exc:
        raise ValueError(f"{name} is not an existing directory") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{name} must be a real non-symlink directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise ValueError(f"{name} could not be opened safely") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode):
        os.close(descriptor)
        raise ValueError(f"{name} changed during open")
    return descriptor


def _read_relative_file_v5(
    root: Path,
    relative_path: str,
    *,
    expected_file_sha256: str,
    name: str,
) -> bytes:
    relative = _canonical_relative_path_v5(relative_path, name=f"{name} path")
    _require_sha256(expected_file_sha256, name=f"{name} file hash")
    root_fd = _open_root_directory_v5(root, name=f"{name} root")
    current_fd = root_fd
    try:
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            next_fd = os.open(part, flags, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
            if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
                raise ValueError(f"{name} parent component is not a directory")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        file_fd = os.open(parts[-1], flags, dir_fd=current_fd)
        try:
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError(f"{name} must be a singly-linked regular file")
            with os.fdopen(file_fd, "rb", closefd=False) as handle:
                encoded = handle.read()
        finally:
            os.close(file_fd)
    except OSError as exc:
        raise ValueError(f"{name} could not be opened safely") from exc
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != expected_file_sha256:
        raise ValueError(f"{name} file hash changed")
    return encoded


def _parse_canonical_json_file_v5(
    encoded: bytes,
    *,
    name: str,
) -> dict[str, Any]:
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    if encoded != _canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} is not canonical JSON")
    claimed = value.get("content_sha256")
    _require_sha256(claimed, name=f"{name} content hash")
    content = dict(value)
    del content["content_sha256"]
    if claimed != _canonical_sha256(content):
        raise ValueError(f"{name} content hash changed")
    return value


def _artifact_spec_v5(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "root",
        "path",
        "file_sha256",
    }:
        raise ValueError(f"authority artifact {name} fields changed")
    root = value.get("root")
    if root not in {"artifact", "registry"}:
        raise ValueError(f"authority artifact {name} root changed")
    path = _canonical_relative_path_v5(value.get("path"), name=f"{name} path")
    file_sha256 = value.get("file_sha256")
    _require_sha256(file_sha256, name=f"{name} file hash")
    return {"root": root, "path": path, "file_sha256": file_sha256}


_COMMON_AUTHORITY_ARTIFACTS_V5 = frozenset(
    {
        "dataset_manifest",
        "corpus_plan",
        "geometry_contract",
        "camera_calibration",
        "implementation",
        "fit_gate_report",
        "v4_fit_checkpoint",
        "training_run",
        "attempt_registry_source",
        "training_access_ledger",
        "g2_role_commitment",
        "g3_role_commitment",
        "g2_evaluation_protocol",
        "g3_evaluation_protocol",
        "g2_finalizer_source",
        "g3_finalizer_source",
    }
)


def _load_authority_manifest_v5(
    context: ProductionCheckpointContextV5,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    encoded = _read_relative_file_v5(
        context.artifact_root,
        context.authority_manifest_path,
        expected_file_sha256=context.authority_manifest_file_sha256,
        name="production authority manifest",
    )
    manifest = _parse_canonical_json_file_v5(
        encoded,
        name="production authority manifest",
    )
    expected_fields = {
        "schema",
        "lifecycle",
        "gate",
        "protocol_generation",
        "artifacts",
        "content_sha256",
    }
    if set(manifest) != expected_fields:
        raise ValueError("production authority manifest fields changed")
    lifecycle = manifest.get("lifecycle")
    gate = manifest.get("gate")
    if (
        manifest.get("schema") != PRODUCTION_AUTHORITY_MANIFEST_V5_SCHEMA
        or lifecycle not in {LIFECYCLE_G3_CANDIDATE, LIFECYCLE_PROMOTED}
        or gate != ("g2" if lifecycle == LIFECYCLE_G3_CANDIDATE else "g3")
        or type(manifest.get("protocol_generation")) is not str
        or not manifest["protocol_generation"]
    ):
        raise ValueError("production authority manifest identity changed")
    expected_artifacts = set(_COMMON_AUTHORITY_ARTIFACTS_V5) | {
        f"{gate}_access_ledger",
        f"{gate}_raw_result",
    }
    if lifecycle == LIFECYCLE_PROMOTED:
        expected_artifacts.update(
            {
                "prior_g2_report",
                "prior_g2_finalized",
                "prior_g2_authority_manifest",
            }
        )
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, Mapping) or set(raw_artifacts) != expected_artifacts:
        raise ValueError("production authority artifact inventory changed")
    artifacts = {
        name: _artifact_spec_v5(raw_artifacts[name], name=name)
        for name in sorted(expected_artifacts)
    }
    for name, spec in artifacts.items():
        expected_root = (
            "registry"
            if name in {"prior_g2_report", "prior_g2_finalized"}
            else "artifact"
        )
        if spec["root"] != expected_root:
            raise ValueError(f"authority artifact {name} uses the wrong root")
    return manifest, artifacts


def _open_authorized_artifacts_v5(
    context: ProductionCheckpointContextV5,
    artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, bytes]:
    # All specs were constrained before this loop. No role/raw payload is opened
    # until the caller has atomically acquired the registry namespace.
    result: dict[str, bytes] = {}
    for name in sorted(artifacts):
        spec = artifacts[name]
        root = (
            context.artifact_root
            if spec["root"] == "artifact"
            else context.registry_root
        )
        result[name] = _read_relative_file_v5(
            root,
            spec["path"],
            expected_file_sha256=spec["file_sha256"],
            name=f"authorized artifact {name}",
        )
    return result


def _exact_string_list_v5(value: object, *, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not str or not item for item in value)
        or value != sorted(value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a nonempty sorted unique string list")
    return tuple(value)


def _validate_dataset_roles_v5(value: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    if set(value) != {"schema", "roles", "content_sha256"} or value.get(
        "schema"
    ) != DATASET_ROLE_MANIFEST_V5_SCHEMA:
        raise ValueError("dataset role manifest fields changed")
    roles = value.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {"train", "g2", "g3"}:
        raise PermissionError("dataset roles must be exactly train/g2/g3")
    normalized = {
        role: _exact_string_list_v5(roles[role], name=f"{role} scene ids")
        for role in ("train", "g2", "g3")
    }
    flattened: list[str] = []
    for role in normalized:
        flattened.extend(normalized[role])
    if len(flattened) != len(set(flattened)):
        raise PermissionError("dataset train/g2/g3 roles are not scene-disjoint")
    if any(
        forbidden in scene_id.lower()
        for scene_id in flattened
        for forbidden in _FORBIDDEN_ROLE_NAMES
    ):
        raise PermissionError("dataset role manifest contains a forbidden role identity")
    return normalized


def _validate_role_commitment_v5(
    value: Mapping[str, Any],
    *,
    gate: str,
    dataset_content_sha256: str,
    protocol_generation: str,
    expected_scene_ids: tuple[str, ...],
) -> None:
    expected_fields = {
        "schema",
        "gate",
        "dataset_manifest_content_sha256",
        "protocol_generation",
        "scene_ids",
        "forbidden_roles",
        "content_sha256",
    }
    if set(value) != expected_fields or value.get("schema") != ROLE_COMMITMENT_V5_SCHEMA:
        raise ValueError(f"{gate} role commitment fields changed")
    if (
        value.get("gate") != gate
        or value.get("dataset_manifest_content_sha256") != dataset_content_sha256
        or value.get("protocol_generation") != protocol_generation
        or tuple(value.get("scene_ids", ())) != expected_scene_ids
        or value.get("forbidden_roles") != list(_FORBIDDEN_ROLE_NAMES)
    ):
        raise PermissionError(f"{gate} role commitment changed")


def _validate_protocol_v5(
    value: Mapping[str, Any],
    *,
    gate: str,
    protocol_generation: str,
) -> None:
    expected_fields = {
        "schema",
        "gate",
        "generation",
        "metric_names",
        "thresholds",
        "content_sha256",
    }
    if set(value) != expected_fields or value.get("schema") != EVALUATION_PROTOCOL_V5_SCHEMA:
        raise ValueError(f"{gate} evaluation protocol fields changed")
    if (
        value.get("gate") != gate
        or value.get("generation") != protocol_generation
        or value.get("metric_names")
        != list(G2_GATE_METRICS_V5 if gate == "g2" else G3_GATE_METRICS_V5)
        or value.get("thresholds") != gate_thresholds_v5(gate)
    ):
        raise PermissionError(f"{gate} evaluation protocol changed")


def _validate_access_ledger_v5(
    value: Mapping[str, Any],
    *,
    role: str,
    expected_scene_ids: tuple[str, ...],
    producer_source_file_sha256: str,
) -> None:
    expected_fields = {
        "schema",
        "role",
        "allowed_scene_ids",
        "opened_scene_ids",
        "forbidden_roles",
        "forbidden_accesses",
        "all_accesses_authorized",
        "producer_source_file_sha256",
        "content_sha256",
    }
    if set(value) != expected_fields or value.get("schema") != ACCESS_LEDGER_V5_SCHEMA:
        raise ValueError(f"{role} access ledger fields changed")
    if (
        value.get("role") != role
        or tuple(value.get("allowed_scene_ids", ())) != expected_scene_ids
        or tuple(value.get("opened_scene_ids", ())) != expected_scene_ids
        or value.get("forbidden_roles") != list(_FORBIDDEN_ROLE_NAMES)
        or value.get("forbidden_accesses") != []
        or value.get("all_accesses_authorized") is not True
        or value.get("producer_source_file_sha256")
        != producer_source_file_sha256
    ):
        raise PermissionError(f"{role} access ledger admits forbidden or missing access")


def _validate_training_run_v5(
    value: Mapping[str, Any],
    *,
    dataset_content_sha256: str,
    training_scene_ids: tuple[str, ...],
    training_access_ledger_content_sha256: str,
    model_state_sha256: str,
) -> None:
    expected_fields = {
        "schema",
        "dataset_manifest_content_sha256",
        "training_scene_ids",
        "training_access_ledger_content_sha256",
        "model_state_sha256",
        "content_sha256",
    }
    if set(value) != expected_fields or value.get("schema") != TRAINING_RUN_V5_SCHEMA:
        raise ValueError("training-run fields changed")
    if (
        value.get("dataset_manifest_content_sha256") != dataset_content_sha256
        or tuple(value.get("training_scene_ids", ())) != training_scene_ids
        or value.get("training_access_ledger_content_sha256")
        != training_access_ledger_content_sha256
        or value.get("model_state_sha256") != model_state_sha256
    ):
        raise PermissionError("training-run provenance changed")


def _validate_raw_gate_result_v5(
    value: Mapping[str, Any],
    *,
    gate: str,
    model_state_sha256: str,
    dataset_content_sha256: str,
    role_commitment_content_sha256: str,
    protocol_content_sha256: str,
    access_ledger_content_sha256: str,
) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, float]]:
    expected_fields = {
        "schema",
        "gate",
        "model_state_sha256",
        "dataset_manifest_content_sha256",
        "role_commitment_content_sha256",
        "evaluation_protocol_content_sha256",
        "access_ledger_content_sha256",
        "families",
        "content_sha256",
    }
    if set(value) != expected_fields or value.get("schema") != RAW_GATE_RESULT_V5_SCHEMA:
        raise ValueError(f"{gate} raw-result fields changed")
    if (
        value.get("gate") != gate
        or value.get("model_state_sha256") != model_state_sha256
        or value.get("dataset_manifest_content_sha256") != dataset_content_sha256
        or value.get("role_commitment_content_sha256")
        != role_commitment_content_sha256
        or value.get("evaluation_protocol_content_sha256")
        != protocol_content_sha256
        or value.get("access_ledger_content_sha256")
        != access_ledger_content_sha256
    ):
        raise PermissionError(f"{gate} raw result has foreign bindings")
    families = value.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError(f"{gate} raw result needs per-family counts")
    metric_names = G2_GATE_METRICS_V5 if gate == "g2" else G3_GATE_METRICS_V5
    normalized: dict[str, dict[str, dict[str, int]]] = {}
    prior_family = ""
    for row in families:
        if not isinstance(row, Mapping) or set(row) != {"family", "metrics"}:
            raise ValueError(f"{gate} raw family fields changed")
        family = row.get("family")
        metrics = row.get("metrics")
        if type(family) is not str or not family or family <= prior_family:
            raise ValueError(f"{gate} raw families must be sorted and unique")
        prior_family = family
        if not isinstance(metrics, Mapping) or set(metrics) != set(metric_names):
            raise ValueError(f"{gate} raw family metric inventory changed")
        normalized[family] = {}
        for metric_name in metric_names:
            count = metrics[metric_name]
            if not isinstance(count, Mapping) or set(count) != {
                "numerator",
                "denominator",
            }:
                raise ValueError(f"{gate} raw metric count fields changed")
            numerator = count.get("numerator")
            denominator = count.get("denominator")
            if (
                isinstance(numerator, bool)
                or isinstance(denominator, bool)
                or not isinstance(numerator, int)
                or not isinstance(denominator, int)
                or denominator <= 0
                or numerator < 0
                or numerator > denominator
            ):
                raise ValueError(f"{gate} raw metric counts are invalid")
            normalized[family][metric_name] = {
                "numerator": numerator,
                "denominator": denominator,
            }
    metrics: dict[str, float] = {}
    for metric_name in metric_names:
        numerator = sum(row[metric_name]["numerator"] for row in normalized.values())
        denominator = sum(
            row[metric_name]["denominator"] for row in normalized.values()
        )
        metrics[metric_name] = numerator / denominator
    return normalized, metrics


def _registry_namespace_v5(
    *,
    gate: str,
    dataset_manifest_file_sha256: str,
    role_commitment_file_sha256: str,
    protocol_generation: str,
) -> str:
    return _canonical_sha256(
        {
            "schema": "lewm_go2_shared_jepa_role_global_registry_namespace_v5",
            "gate": gate,
            "dataset_manifest_file_sha256": dataset_manifest_file_sha256,
            "role_commitment_file_sha256": role_commitment_file_sha256,
            "protocol_generation": protocol_generation,
        }
    )


def _write_exclusive_at_v5(directory_fd: int, name: str, encoded: bytes) -> str:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    return hashlib.sha256(encoded).hexdigest()


def _acquire_gate_reservation_v5(
    context: ProductionCheckpointContextV5,
    *,
    gate: str,
    namespace_sha256: str,
    reservation: Mapping[str, Any],
) -> tuple[str, str]:
    registry_fd = _open_root_directory_v5(context.registry_root, name="registry_root")
    gate_fd: int | None = None
    namespace_fd: int | None = None
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        gate_fd = os.open(gate, flags, dir_fd=registry_fd)
        try:
            os.mkdir(namespace_sha256, mode=0o700, dir_fd=gate_fd)
        except FileExistsError as exc:
            raise PermissionError(
                f"{gate} role-global attempt was already reserved; retries/copies are forbidden"
            ) from exc
        namespace_fd = os.open(namespace_sha256, flags, dir_fd=gate_fd)
        reservation_payload = dict(reservation)
        reservation_payload["content_sha256"] = _canonical_sha256(reservation_payload)
        encoded = _canonical_json_bytes(reservation_payload) + b"\n"
        file_sha256 = _write_exclusive_at_v5(
            namespace_fd,
            "reservation.json",
            encoded,
        )
        os.fsync(namespace_fd)
        os.fsync(gate_fd)
        return reservation_payload["content_sha256"], file_sha256
    finally:
        if namespace_fd is not None:
            os.close(namespace_fd)
        if gate_fd is not None:
            os.close(gate_fd)
        os.close(registry_fd)


@dataclass(frozen=True)
class _AuthorizedGateInputsV5:
    manifest: dict[str, Any]
    artifacts: dict[str, dict[str, str]]
    parsed: dict[str, dict[str, Any]]
    roles: dict[str, tuple[str, ...]]
    provenance: CheckpointProvenanceV5
    per_family_counts: dict[str, dict[str, dict[str, int]]]
    metrics: dict[str, float]


def _structured_artifact_v5(
    encoded: Mapping[str, bytes],
    name: str,
) -> dict[str, Any]:
    return _parse_canonical_json_file_v5(
        encoded[name],
        name=f"authorized artifact {name}",
    )


def _validate_authorized_gate_inputs_v5(
    *,
    manifest: dict[str, Any],
    artifacts: dict[str, dict[str, str]],
    encoded: dict[str, bytes],
    model_state_sha256: str,
) -> _AuthorizedGateInputsV5:
    gate = manifest["gate"]
    generation = manifest["protocol_generation"]
    current_source = Path(__file__).read_bytes()
    for source_name in (
        "implementation",
        "attempt_registry_source",
        "g2_finalizer_source",
        "g3_finalizer_source",
    ):
        if encoded[source_name] != current_source:
            raise PermissionError(
                f"authorized {source_name} is not the executing V5 implementation"
            )
    parsed_names = {
        "dataset_manifest",
        "training_run",
        "training_access_ledger",
        "g2_role_commitment",
        "g3_role_commitment",
        "g2_evaluation_protocol",
        "g3_evaluation_protocol",
        f"{gate}_access_ledger",
        f"{gate}_raw_result",
    }
    if "prior_g2_report" in encoded:
        parsed_names.add("prior_g2_report")
        parsed_names.add("prior_g2_authority_manifest")
    parsed = {
        name: _structured_artifact_v5(encoded, name)
        for name in sorted(parsed_names)
    }
    dataset = parsed["dataset_manifest"]
    roles = _validate_dataset_roles_v5(dataset)
    dataset_content_sha256 = dataset["content_sha256"]
    registry_source_hash = artifacts["attempt_registry_source"]["file_sha256"]
    training_ledger = parsed["training_access_ledger"]
    _validate_access_ledger_v5(
        training_ledger,
        role="train",
        expected_scene_ids=roles["train"],
        producer_source_file_sha256=registry_source_hash,
    )
    for current_gate in ("g2", "g3"):
        role = parsed[f"{current_gate}_role_commitment"]
        protocol = parsed[f"{current_gate}_evaluation_protocol"]
        _validate_role_commitment_v5(
            role,
            gate=current_gate,
            dataset_content_sha256=dataset_content_sha256,
            protocol_generation=generation,
            expected_scene_ids=roles[current_gate],
        )
        _validate_protocol_v5(
            protocol,
            gate=current_gate,
            protocol_generation=generation,
        )
    gate_ledger = parsed[f"{gate}_access_ledger"]
    _validate_access_ledger_v5(
        gate_ledger,
        role=gate,
        expected_scene_ids=roles[gate],
        producer_source_file_sha256=registry_source_hash,
    )
    _validate_training_run_v5(
        parsed["training_run"],
        dataset_content_sha256=dataset_content_sha256,
        training_scene_ids=roles["train"],
        training_access_ledger_content_sha256=training_ledger["content_sha256"],
        model_state_sha256=model_state_sha256,
    )
    raw = parsed[f"{gate}_raw_result"]
    per_family_counts, metrics = _validate_raw_gate_result_v5(
        raw,
        gate=gate,
        model_state_sha256=model_state_sha256,
        dataset_content_sha256=dataset_content_sha256,
        role_commitment_content_sha256=parsed[f"{gate}_role_commitment"][
            "content_sha256"
        ],
        protocol_content_sha256=parsed[f"{gate}_evaluation_protocol"][
            "content_sha256"
        ],
        access_ledger_content_sha256=gate_ledger["content_sha256"],
    )
    provenance = CheckpointProvenanceV5(
        dataset_manifest_sha256=artifacts["dataset_manifest"]["file_sha256"],
        corpus_plan_sha256=artifacts["corpus_plan"]["file_sha256"],
        geometry_contract_sha256=artifacts["geometry_contract"]["file_sha256"],
        camera_calibration_sha256=artifacts["camera_calibration"]["file_sha256"],
        implementation_sha256=artifacts["implementation"]["file_sha256"],
        fit_gate_report_sha256=artifacts["fit_gate_report"]["file_sha256"],
        v4_fit_checkpoint_sha256=artifacts["v4_fit_checkpoint"]["file_sha256"],
        training_run_sha256=artifacts["training_run"]["file_sha256"],
        gate_attempt_registry_source_sha256=registry_source_hash,
        g2_role_commitment_sha256=artifacts["g2_role_commitment"]["file_sha256"],
        g3_role_commitment_sha256=artifacts["g3_role_commitment"]["file_sha256"],
        g2_evaluation_protocol_sha256=artifacts["g2_evaluation_protocol"][
            "file_sha256"
        ],
        g3_evaluation_protocol_sha256=artifacts["g3_evaluation_protocol"][
            "file_sha256"
        ],
        g2_finalizer_source_sha256=artifacts["g2_finalizer_source"]["file_sha256"],
        g3_finalizer_source_sha256=artifacts["g3_finalizer_source"]["file_sha256"],
        training_scene_ids=roles["train"],
    )
    return _AuthorizedGateInputsV5(
        manifest=manifest,
        artifacts=artifacts,
        parsed=parsed,
        roles=roles,
        provenance=provenance,
        per_family_counts=per_family_counts,
        metrics=metrics,
    )


def _secure_gate_report_v5(
    *,
    gate: str,
    model: SharedObservableCameraRayJepaV5,
    inputs: _AuthorizedGateInputsV5,
    namespace_sha256: str,
    reservation_content_sha256: str,
    reservation_file_sha256: str,
    authority_manifest_file_sha256: str,
) -> dict[str, Any]:
    thresholds = gate_thresholds_v5(gate)
    checks = {
        name: inputs.metrics[name] >= float(thresholds["metrics"][name]["value"])
        for name in sorted(inputs.metrics)
    }
    passed = all(checks.values())
    metrics_sha256 = _canonical_sha256(inputs.metrics)
    thresholds_sha256 = _canonical_sha256(thresholds)
    decision_sha256 = _canonical_sha256(
        {
            "schema": f"lewm_go2_shared_jepa_{gate}_decision_v5",
            "metrics_sha256": metrics_sha256,
            "thresholds_sha256": thresholds_sha256,
            "checks": checks,
            "passed": passed,
        }
    )
    bindings = checkpoint_contract_bindings_v5(model, inputs.provenance)
    raw = inputs.parsed[f"{gate}_raw_result"]
    ledger = inputs.parsed[f"{gate}_access_ledger"]
    attempt_registry = {
        "schema": f"lewm_go2_shared_jepa_{gate}_filesystem_registry_receipt_v5",
        "gate": gate,
        "status": "role_global_filesystem_finalized",
        "namespace_sha256": namespace_sha256,
        "reservation_content_sha256": reservation_content_sha256,
        "reservation_file_sha256": reservation_file_sha256,
        "authority_manifest_file_sha256": authority_manifest_file_sha256,
        "dataset_manifest_sha256": inputs.provenance.dataset_manifest_sha256,
        "gate_role_commitment_sha256": getattr(
            inputs.provenance, f"{gate}_role_commitment_sha256"
        ),
        "evaluation_protocol_sha256": getattr(
            inputs.provenance, f"{gate}_evaluation_protocol_sha256"
        ),
        "threshold_contract_sha256": thresholds_sha256,
        "model_state_sha256": bindings["model_state_sha256"],
        "implementation_sha256": inputs.provenance.implementation_sha256,
        "raw_result_file_sha256": inputs.artifacts[f"{gate}_raw_result"][
            "file_sha256"
        ],
    }
    attempt_registry["content_sha256"] = _canonical_sha256(attempt_registry)
    finalizer = {
        "schema": f"lewm_go2_shared_jepa_{gate}_finalizer_v5",
        "gate": gate,
        "status": "independently_reconstructed_from_raw_counts",
        "finalizer_source_sha256": getattr(
            inputs.provenance, f"{gate}_finalizer_source_sha256"
        ),
        "implementation_sha256": inputs.provenance.implementation_sha256,
        "attempt_registry_content_sha256": attempt_registry["content_sha256"],
        "raw_result_content_sha256": raw["content_sha256"],
        "raw_result_file_sha256": inputs.artifacts[f"{gate}_raw_result"][
            "file_sha256"
        ],
        "access_ledger_content_sha256": ledger["content_sha256"],
        "metrics_sha256": metrics_sha256,
        "thresholds_sha256": thresholds_sha256,
        "decision_sha256": decision_sha256,
    }
    finalizer["content_sha256"] = _canonical_sha256(finalizer)
    report = {
        "schema": _gate_report_schema_v5(gate),
        "gate": gate,
        "passed": passed,
        "model_family": MODEL_FAMILY,
        "metrics": inputs.metrics,
        "thresholds": thresholds,
        "checks": checks,
        "decision_sha256": decision_sha256,
        "per_family_counts": inputs.per_family_counts,
        "raw_result": raw,
        "access_ledger": ledger,
        "attempt_registry": attempt_registry,
        "finalizer": finalizer,
        **bindings,
    }
    report["content_sha256"] = _canonical_sha256(report)
    if not passed:
        raise PermissionError(f"{gate} raw counts do not pass frozen thresholds")
    return report


def _finalize_g2_raw_result_v5(**kwargs: Any) -> dict[str, Any]:
    if kwargs.get("gate") != "g2":
        raise ValueError("G2 finalizer received another gate")
    return _secure_gate_report_v5(**kwargs)


def _finalize_g3_raw_result_v5(**kwargs: Any) -> dict[str, Any]:
    if kwargs.get("gate") != "g3":
        raise ValueError("G3 finalizer received another gate")
    return _secure_gate_report_v5(**kwargs)


def _write_gate_finalization_v5(
    context: ProductionCheckpointContextV5,
    *,
    gate: str,
    namespace_sha256: str,
    report: Mapping[str, Any],
    reservation_content_sha256: str,
    reservation_file_sha256: str,
) -> tuple[str, str, str]:
    registry_fd = _open_root_directory_v5(context.registry_root, name="registry_root")
    gate_fd: int | None = None
    namespace_fd: int | None = None
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        gate_fd = os.open(gate, flags, dir_fd=registry_fd)
        namespace_fd = os.open(namespace_sha256, flags, dir_fd=gate_fd)
        report_encoded = _canonical_json_bytes(dict(report)) + b"\n"
        report_file_sha256 = _write_exclusive_at_v5(
            namespace_fd,
            "gate_report.json",
            report_encoded,
        )
        finalized = {
            "schema": f"lewm_go2_shared_jepa_{gate}_filesystem_finalization_v5",
            "gate": gate,
            "namespace_sha256": namespace_sha256,
            "reservation_content_sha256": reservation_content_sha256,
            "reservation_file_sha256": reservation_file_sha256,
            "gate_report_content_sha256": report["content_sha256"],
            "gate_report_file_sha256": report_file_sha256,
            "model_state_sha256": report["model_state_sha256"],
            "raw_result_file_sha256": report["attempt_registry"][
                "raw_result_file_sha256"
            ],
        }
        finalized["content_sha256"] = _canonical_sha256(finalized)
        finalized_encoded = _canonical_json_bytes(finalized) + b"\n"
        finalized_file_sha256 = _write_exclusive_at_v5(
            namespace_fd,
            "finalized.json",
            finalized_encoded,
        )
        os.fsync(namespace_fd)
        os.fsync(gate_fd)
        relative_report = f"{gate}/{namespace_sha256}/gate_report.json"
        return relative_report, report_file_sha256, finalized_file_sha256
    finally:
        if namespace_fd is not None:
            os.close(namespace_fd)
        if gate_fd is not None:
            os.close(gate_fd)
        os.close(registry_fd)


def _validate_secure_gate_report_v5(
    report: object,
    *,
    gate: str,
    model: SharedObservableCameraRayJepaV5,
    inputs: _AuthorizedGateInputsV5,
    context: ProductionCheckpointContextV5,
    report_path: str,
    report_file_sha256: str,
    finalized_file_sha256: str,
    expected_authority_manifest_file_sha256: str,
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "gate",
        "passed",
        "model_family",
        "metrics",
        "thresholds",
        "checks",
        "decision_sha256",
        "per_family_counts",
        "raw_result",
        "access_ledger",
        "attempt_registry",
        "finalizer",
        "content_sha256",
        *checkpoint_contract_bindings_v5(model, inputs.provenance),
    }
    if not isinstance(report, Mapping) or set(report) != expected_fields:
        raise ValueError(f"{gate} secure gate-report fields changed")
    normalized = json.loads(_canonical_json_bytes(dict(report)).decode("utf-8"))
    claimed = normalized.get("content_sha256")
    content = dict(normalized)
    content.pop("content_sha256", None)
    if claimed != _canonical_sha256(content):
        raise ValueError(f"{gate} secure gate-report content hash changed")
    if (
        normalized.get("schema") != _gate_report_schema_v5(gate)
        or normalized.get("gate") != gate
        or normalized.get("model_family") != MODEL_FAMILY
        or normalized.get("passed") is not True
    ):
        raise PermissionError(f"{gate} secure report does not record a pass")
    bindings = checkpoint_contract_bindings_v5(model, inputs.provenance)
    for name, expected in bindings.items():
        if normalized.get(name) != expected:
            raise PermissionError(f"{gate} secure report has foreign {name}")
    raw = normalized.get("raw_result")
    ledger = normalized.get("access_ledger")
    if not isinstance(raw, Mapping) or not isinstance(ledger, Mapping):
        raise ValueError(f"{gate} secure report lost raw result or access ledger")
    for value, name in ((raw, "raw result"), (ledger, "access ledger")):
        claimed_nested = value.get("content_sha256")
        nested = dict(value)
        nested.pop("content_sha256", None)
        if claimed_nested != _canonical_sha256(nested):
            raise ValueError(f"{gate} embedded {name} content hash changed")
    if (
        raw != inputs.parsed.get(f"{gate}_raw_result")
        or ledger != inputs.parsed.get(f"{gate}_access_ledger")
    ):
        raise PermissionError(
            f"{gate} embedded evidence differs from the reopened raw files"
        )
    _validate_access_ledger_v5(
        ledger,
        role=gate,
        expected_scene_ids=inputs.roles[gate],
        producer_source_file_sha256=(
            inputs.provenance.gate_attempt_registry_source_sha256
        ),
    )
    counts, metrics = _validate_raw_gate_result_v5(
        raw,
        gate=gate,
        model_state_sha256=bindings["model_state_sha256"],
        dataset_content_sha256=inputs.parsed["dataset_manifest"]["content_sha256"],
        role_commitment_content_sha256=inputs.parsed[f"{gate}_role_commitment"][
            "content_sha256"
        ],
        protocol_content_sha256=inputs.parsed[f"{gate}_evaluation_protocol"][
            "content_sha256"
        ],
        access_ledger_content_sha256=ledger["content_sha256"],
    )
    thresholds = gate_thresholds_v5(gate)
    checks = {
        name: metrics[name] >= float(thresholds["metrics"][name]["value"])
        for name in sorted(metrics)
    }
    decision_sha256 = _canonical_sha256(
        {
            "schema": f"lewm_go2_shared_jepa_{gate}_decision_v5",
            "metrics_sha256": _canonical_sha256(metrics),
            "thresholds_sha256": _canonical_sha256(thresholds),
            "checks": checks,
            "passed": all(checks.values()),
        }
    )
    if (
        normalized.get("per_family_counts") != counts
        or normalized.get("metrics") != metrics
        or normalized.get("thresholds") != thresholds
        or normalized.get("checks") != checks
        or not all(checks.values())
        or normalized.get("decision_sha256") != decision_sha256
    ):
        raise PermissionError(f"{gate} secure report was not reconstructed from raw counts")
    attempt = normalized.get("attempt_registry")
    finalizer = normalized.get("finalizer")
    if not isinstance(attempt, Mapping) or not isinstance(finalizer, Mapping):
        raise ValueError(f"{gate} secure registry/finalizer receipt changed")
    namespace = attempt.get("namespace_sha256")
    _require_sha256(namespace, name=f"{gate} registry namespace")
    expected_namespace = _registry_namespace_v5(
        gate=gate,
        dataset_manifest_file_sha256=inputs.provenance.dataset_manifest_sha256,
        role_commitment_file_sha256=getattr(
            inputs.provenance, f"{gate}_role_commitment_sha256"
        ),
        protocol_generation=inputs.manifest["protocol_generation"],
    )
    if namespace != expected_namespace:
        raise PermissionError(f"{gate} registry namespace changed")
    expected_report_path = f"{gate}/{namespace}/gate_report.json"
    if _canonical_relative_path_v5(report_path, name=f"{gate} report path") != expected_report_path:
        raise PermissionError(f"{gate} report path is outside its registry namespace")
    reservation_path = f"{gate}/{namespace}/reservation.json"
    finalized_path = f"{gate}/{namespace}/finalized.json"
    reservation_encoded = _read_relative_file_v5(
        context.registry_root,
        reservation_path,
        expected_file_sha256=attempt.get("reservation_file_sha256"),
        name=f"{gate} filesystem reservation",
    )
    reservation = _parse_canonical_json_file_v5(
        reservation_encoded,
        name=f"{gate} filesystem reservation",
    )
    if (
        reservation.get("content_sha256")
        != attempt.get("reservation_content_sha256")
        or reservation.get("namespace_sha256") != namespace
        or reservation.get("model_state_sha256") != bindings["model_state_sha256"]
        or reservation.get("raw_result_file_sha256")
        != attempt.get("raw_result_file_sha256")
        or reservation.get("implementation_sha256")
        != inputs.provenance.implementation_sha256
        or reservation.get("threshold_contract_sha256")
        != _canonical_sha256(thresholds)
    ):
        raise PermissionError(f"{gate} filesystem reservation binding changed")
    actual_report_encoded = _read_relative_file_v5(
        context.registry_root,
        expected_report_path,
        expected_file_sha256=report_file_sha256,
        name=f"{gate} finalized gate report",
    )
    if actual_report_encoded != _canonical_json_bytes(normalized) + b"\n":
        raise ValueError(f"{gate} embedded report differs from filesystem finalization")
    finalized_encoded = _read_relative_file_v5(
        context.registry_root,
        finalized_path,
        expected_file_sha256=finalized_file_sha256,
        name=f"{gate} finalization record",
    )
    finalized = _parse_canonical_json_file_v5(
        finalized_encoded,
        name=f"{gate} finalization record",
    )
    if (
        finalized.get("gate_report_content_sha256") != claimed
        or finalized.get("gate_report_file_sha256") != report_file_sha256
        or finalized.get("reservation_content_sha256")
        != reservation["content_sha256"]
        or finalized.get("model_state_sha256") != bindings["model_state_sha256"]
        or finalized.get("raw_result_file_sha256")
        != attempt.get("raw_result_file_sha256")
    ):
        raise PermissionError(f"{gate} finalization record binding changed")
    expected_attempt_fields = {
        "schema",
        "gate",
        "status",
        "namespace_sha256",
        "reservation_content_sha256",
        "reservation_file_sha256",
        "authority_manifest_file_sha256",
        "dataset_manifest_sha256",
        "gate_role_commitment_sha256",
        "evaluation_protocol_sha256",
        "threshold_contract_sha256",
        "model_state_sha256",
        "implementation_sha256",
        "raw_result_file_sha256",
        "content_sha256",
    }
    if (
        set(attempt) != expected_attempt_fields
        or attempt.get("status") != "role_global_filesystem_finalized"
        or attempt.get("authority_manifest_file_sha256")
        != expected_authority_manifest_file_sha256
    ):
        raise PermissionError(f"{gate} registry receipt fields changed")
    attempt_content = dict(attempt)
    attempt_claimed = attempt_content.pop("content_sha256", None)
    if attempt_claimed != _canonical_sha256(attempt_content):
        raise ValueError(f"{gate} registry receipt content hash changed")
    expected_finalizer_fields = {
        "schema",
        "gate",
        "status",
        "finalizer_source_sha256",
        "implementation_sha256",
        "attempt_registry_content_sha256",
        "raw_result_content_sha256",
        "raw_result_file_sha256",
        "access_ledger_content_sha256",
        "metrics_sha256",
        "thresholds_sha256",
        "decision_sha256",
        "content_sha256",
    }
    if (
        set(finalizer) != expected_finalizer_fields
        or finalizer.get("status")
        != "independently_reconstructed_from_raw_counts"
        or finalizer.get("finalizer_source_sha256")
        != getattr(inputs.provenance, f"{gate}_finalizer_source_sha256")
        or finalizer.get("raw_result_content_sha256") != raw["content_sha256"]
        or finalizer.get("metrics_sha256") != _canonical_sha256(metrics)
        or finalizer.get("decision_sha256") != decision_sha256
    ):
        raise PermissionError(f"{gate} finalizer receipt binding changed")
    finalizer_content = dict(finalizer)
    finalizer_claimed = finalizer_content.pop("content_sha256", None)
    if finalizer_claimed != _canonical_sha256(finalizer_content):
        raise ValueError(f"{gate} finalizer receipt content hash changed")
    return normalized


def _load_prior_g2_inputs_v5(
    *,
    context: ProductionCheckpointContextV5,
    current_inputs: _AuthorizedGateInputsV5,
    model_state_sha256: str,
) -> tuple[ProductionCheckpointContextV5, _AuthorizedGateInputsV5]:
    spec = current_inputs.artifacts.get("prior_g2_authority_manifest")
    if not isinstance(spec, Mapping) or spec.get("root") != "artifact":
        raise PermissionError("promoted V5 evidence lacks prior G2 authority")
    prior_context = ProductionCheckpointContextV5(
        artifact_root=context.artifact_root,
        registry_root=context.registry_root,
        authority_manifest_path=str(spec["path"]),
        authority_manifest_file_sha256=str(spec["file_sha256"]),
    )
    prior_manifest, prior_artifacts = _load_authority_manifest_v5(prior_context)
    if (
        prior_manifest.get("lifecycle") != LIFECYCLE_G3_CANDIDATE
        or prior_manifest.get("gate") != "g2"
    ):
        raise PermissionError("prior G2 authority manifest changed")
    prior_encoded = _open_authorized_artifacts_v5(prior_context, prior_artifacts)
    prior_inputs = _validate_authorized_gate_inputs_v5(
        manifest=prior_manifest,
        artifacts=prior_artifacts,
        encoded=prior_encoded,
        model_state_sha256=model_state_sha256,
    )
    if prior_inputs.provenance != current_inputs.provenance:
        raise PermissionError("G2/G3 source, dataset, protocol, or state identity changed")
    return prior_context, prior_inputs


_SECURE_CHECKPOINT_EXTRA_FIELDS_V5 = {
    "authority_manifest_path",
    "authority_manifest_file_sha256",
    "authority_manifest_content_sha256",
    "registry_namespace_sha256",
    "g2_report_path",
    "g2_registry_finalized_file_sha256",
    "g3_report_path",
    "g3_registry_finalized_file_sha256",
}


def _secure_checkpoint_fields_v5() -> set[str]:
    return {
        "schema",
        "lifecycle",
        "model_family",
        "runtime_ready",
        "model_state_sha256",
        "model_state_dict",
        "model_config",
        "model_config_sha256",
        "architecture_contract",
        "architecture_contract_sha256",
        "output_contract",
        "output_contract_sha256",
        "v4_geometry",
        "v4_geometry_sha256",
        "bev_geometry",
        "bev_geometry_sha256",
        "provenance",
        "provenance_contract_sha256",
        "g2_report",
        "g2_report_content_sha256",
        "g2_report_source",
        "g2_report_file_sha256",
        "g3_report",
        "g3_report_content_sha256",
        "g3_report_source",
        "g3_report_file_sha256",
        *_SECURE_CHECKPOINT_EXTRA_FIELDS_V5,
    }


def _secure_checkpoint_core_v5(
    payload: Mapping[str, Any],
    *,
    expected_lifecycle: str | None,
) -> tuple[
    SharedObservableCameraRayJepaV5,
    CheckpointProvenanceV5,
    dict[str, str],
]:
    if not isinstance(payload, Mapping) or set(payload) != _secure_checkpoint_fields_v5():
        raise ValueError("secure checkpoint-v5 fields changed")
    lifecycle = payload.get("lifecycle")
    if expected_lifecycle is not None and lifecycle != expected_lifecycle:
        raise PermissionError("checkpoint-v5 lifecycle does not match loader mode")
    if lifecycle not in {LIFECYCLE_G3_CANDIDATE, LIFECYCLE_PROMOTED}:
        raise ValueError("checkpoint-v5 lifecycle is invalid")
    if (
        payload.get("schema") != CHECKPOINT_V5_SCHEMA
        or payload.get("model_family") != MODEL_FAMILY
        or payload.get("runtime_ready") is not (lifecycle == LIFECYCLE_PROMOTED)
    ):
        raise PermissionError("secure checkpoint identity/readiness changed")
    config = SharedObservableCameraRayJepaV5Config.from_mapping(
        _require_mapping(payload.get("model_config"), "model_config")
    )
    _require_production_checkpoint_config(config)
    if payload.get("model_config_sha256") != config.content_sha256:
        raise ValueError("secure checkpoint model-config hash changed")
    model = SharedObservableCameraRayJepaV5(config)
    expected_output = shared_output_contract_v5(model)
    expected_v4 = canonical_v4_geometry_contract_v5(config)
    expected_bev = canonical_bev_geometry_contract_v5(config)
    expected_architecture = shared_architecture_contract_v5(model)
    for name, expected in (
        ("output_contract", expected_output),
        ("v4_geometry", expected_v4),
        ("bev_geometry", expected_bev),
        ("architecture_contract", expected_architecture),
    ):
        _require_canonical_mapping(payload.get(name), expected, name=name)
        if payload.get(f"{name}_sha256") != _canonical_sha256(expected):
            raise ValueError(f"secure checkpoint {name} hash changed")
    state = _require_mapping(payload.get("model_state_dict"), "model_state_dict")
    _validate_state_contract(state, model.deployment_state_dict())
    state_sha256 = tensor_state_dict_sha256(state)
    if payload.get("model_state_sha256") != state_sha256:
        raise ValueError("secure checkpoint model-state hash changed")
    model.load_deployment_state_dict(state)
    if tensor_state_dict_sha256(model.deployment_state_dict()) != state_sha256:
        raise ValueError("secure checkpoint state load changed bytes")
    provenance = _validate_checkpoint_provenance(payload.get("provenance"))
    if payload.get("provenance_contract_sha256") != provenance.content_sha256:
        raise ValueError("secure checkpoint provenance hash changed")
    bindings = checkpoint_contract_bindings_v5(model, provenance)
    return model, provenance, bindings


def _removed_caller_filesystem_validate_checkpoint_v5_payload(
    payload: Mapping[str, Any],
    *,
    expected_lifecycle: str | None = None,
    context: ProductionCheckpointContextV5 | None = None,
) -> None:
    """Reopen and recompute every filesystem-backed promotion claim."""

    raise PermissionError("caller-root checkpoint validation was permanently removed")

    if not isinstance(context, ProductionCheckpointContextV5):
        raise PermissionError(
            "production checkpoint validation requires canonical filesystem evidence"
        )
    model, payload_provenance, bindings = _secure_checkpoint_core_v5(
        payload,
        expected_lifecycle=expected_lifecycle,
    )
    lifecycle = str(payload["lifecycle"])
    manifest, artifacts = _load_authority_manifest_v5(context)
    if (
        manifest.get("lifecycle") != lifecycle
        or payload.get("authority_manifest_path") != context.authority_manifest_path
        or payload.get("authority_manifest_file_sha256")
        != context.authority_manifest_file_sha256
        or payload.get("authority_manifest_content_sha256")
        != manifest.get("content_sha256")
    ):
        raise PermissionError("checkpoint authority-manifest binding changed")
    encoded = _open_authorized_artifacts_v5(context, artifacts)
    inputs = _validate_authorized_gate_inputs_v5(
        manifest=manifest,
        artifacts=artifacts,
        encoded=encoded,
        model_state_sha256=bindings["model_state_sha256"],
    )
    if inputs.provenance != payload_provenance:
        raise PermissionError("checkpoint filesystem provenance changed")
    gate = manifest["gate"]
    namespace = _registry_namespace_v5(
        gate=gate,
        dataset_manifest_file_sha256=inputs.provenance.dataset_manifest_sha256,
        role_commitment_file_sha256=getattr(
            inputs.provenance, f"{gate}_role_commitment_sha256"
        ),
        protocol_generation=manifest["protocol_generation"],
    )
    if payload.get("registry_namespace_sha256") != namespace:
        raise PermissionError("checkpoint role-global registry namespace changed")

    def validate_report(
        *,
        report_gate: str,
        report_inputs: _AuthorizedGateInputsV5,
        report_context: ProductionCheckpointContextV5,
        report_path: object,
        report_file_sha256: object,
        finalized_file_sha256: object,
        expected_authority_file_sha256: str,
    ) -> dict[str, Any]:
        if not all(
            isinstance(value, str)
            for value in (report_path, report_file_sha256, finalized_file_sha256)
        ):
            raise PermissionError(f"{report_gate} checkpoint file bindings are missing")
        return _validate_secure_gate_report_v5(
            payload[f"{report_gate}_report"],
            gate=report_gate,
            model=model,
            inputs=report_inputs,
            context=report_context,
            report_path=str(report_path),
            report_file_sha256=str(report_file_sha256),
            finalized_file_sha256=str(finalized_file_sha256),
            expected_authority_manifest_file_sha256=(
                expected_authority_file_sha256
            ),
        )

    if lifecycle == LIFECYCLE_G3_CANDIDATE:
        if gate != "g2":
            raise PermissionError("G3 candidate authority must finalize G2")
        g2 = validate_report(
            report_gate="g2",
            report_inputs=inputs,
            report_context=context,
            report_path=payload.get("g2_report_path"),
            report_file_sha256=payload.get("g2_report_file_sha256"),
            finalized_file_sha256=payload.get(
                "g2_registry_finalized_file_sha256"
            ),
            expected_authority_file_sha256=context.authority_manifest_file_sha256,
        )
        if (
            payload.get("g2_report") != g2
            or payload.get("g2_report_content_sha256") != g2["content_sha256"]
            or payload.get("g2_report_source") != _GATE_REPORT_REGISTRY_SOURCE
            or any(
                payload.get(name) is not None
                for name in (
                    "g3_report",
                    "g3_report_content_sha256",
                    "g3_report_source",
                    "g3_report_file_sha256",
                    "g3_report_path",
                    "g3_registry_finalized_file_sha256",
                )
            )
        ):
            raise PermissionError("G3 candidate checkpoint evidence changed")
        return

    if gate != "g3":
        raise PermissionError("promoted authority must finalize G3")
    prior_context, prior_inputs = _load_prior_g2_inputs_v5(
        context=context,
        current_inputs=inputs,
        model_state_sha256=bindings["model_state_sha256"],
    )
    prior_authority_spec = artifacts["prior_g2_authority_manifest"]
    g2 = validate_report(
        report_gate="g2",
        report_inputs=prior_inputs,
        report_context=prior_context,
        report_path=payload.get("g2_report_path"),
        report_file_sha256=payload.get("g2_report_file_sha256"),
        finalized_file_sha256=payload.get("g2_registry_finalized_file_sha256"),
        expected_authority_file_sha256=prior_authority_spec["file_sha256"],
    )
    g3 = validate_report(
        report_gate="g3",
        report_inputs=inputs,
        report_context=context,
        report_path=payload.get("g3_report_path"),
        report_file_sha256=payload.get("g3_report_file_sha256"),
        finalized_file_sha256=payload.get("g3_registry_finalized_file_sha256"),
        expected_authority_file_sha256=context.authority_manifest_file_sha256,
    )
    if (
        payload.get("g2_report") != g2
        or payload.get("g2_report_content_sha256") != g2["content_sha256"]
        or payload.get("g2_report_source") != _GATE_REPORT_REGISTRY_SOURCE
        or payload.get("g3_report") != g3
        or payload.get("g3_report_content_sha256") != g3["content_sha256"]
        or payload.get("g3_report_source") != _GATE_REPORT_REGISTRY_SOURCE
    ):
        raise PermissionError("promoted checkpoint filesystem evidence changed")


def _removed_caller_filesystem_checkpoint_v5_weights_only_roundtrip(
    payload: Mapping[str, Any],
    *,
    expected_lifecycle: str | None = None,
    context: ProductionCheckpointContextV5 | None = None,
) -> dict[str, Any]:
    raise PermissionError("caller-root checkpoint roundtrip was permanently removed")
    if not isinstance(context, ProductionCheckpointContextV5):
        raise PermissionError("weights-only roundtrip requires filesystem evidence")
    validate_checkpoint_v5_payload(
        payload,
        expected_lifecycle=expected_lifecycle,
        context=context,
    )
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)
    if not isinstance(restored, dict):
        raise ValueError("checkpoint-v5 weights-only roundtrip changed root type")
    validate_checkpoint_v5_payload(
        restored,
        expected_lifecycle=expected_lifecycle,
        context=context,
    )
    return restored


def _reservation_payload_v5(
    *,
    gate: str,
    namespace_sha256: str,
    context: ProductionCheckpointContextV5,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, str]],
    model_state_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": f"lewm_go2_shared_jepa_{gate}_filesystem_reservation_v5",
        "gate": gate,
        "status": "reserved_consumes_role_global_attempt",
        "namespace_sha256": namespace_sha256,
        "authority_manifest_file_sha256": context.authority_manifest_file_sha256,
        "authority_manifest_content_sha256": manifest["content_sha256"],
        "dataset_manifest_file_sha256": artifacts["dataset_manifest"][
            "file_sha256"
        ],
        "role_commitment_file_sha256": artifacts[f"{gate}_role_commitment"][
            "file_sha256"
        ],
        "evaluation_protocol_file_sha256": artifacts[
            f"{gate}_evaluation_protocol"
        ]["file_sha256"],
        "protocol_generation": manifest["protocol_generation"],
        "threshold_contract_sha256": _canonical_sha256(
            gate_thresholds_v5(gate)
        ),
        "model_state_sha256": model_state_sha256,
        "implementation_sha256": artifacts["implementation"]["file_sha256"],
        "raw_result_file_sha256": artifacts[f"{gate}_raw_result"][
            "file_sha256"
        ],
        "attempt_registry_source_sha256": artifacts[
            "attempt_registry_source"
        ]["file_sha256"],
        "created_at_ns": time.time_ns(),
    }


def _removed_caller_filesystem_build_checkpoint_v5_payload(
    model: SharedObservableCameraRayJepaV5,
    *,
    lifecycle: str,
    context: ProductionCheckpointContextV5,
) -> dict[str, Any]:
    """Build only from pre-authorized files and a newly acquired one-shot role."""

    raise PermissionError("caller-root checkpoint construction was permanently removed")

    if not isinstance(model, SharedObservableCameraRayJepaV5):
        raise TypeError("model must be SharedObservableCameraRayJepaV5")
    if not isinstance(context, ProductionCheckpointContextV5):
        raise TypeError("production checkpoint context is required")
    if lifecycle not in {LIFECYCLE_G3_CANDIDATE, LIFECYCLE_PROMOTED}:
        raise ValueError("checkpoint lifecycle must be g3_candidate or promoted")
    _require_production_checkpoint_config(model.model_config)
    state = model.deployment_state_dict()
    _validate_state_contract(state, model.deployment_state_dict())
    state_sha256 = tensor_state_dict_sha256(state)

    # The authority manifest is pre-authorization metadata. It is the only file
    # opened before reservation; every declared role/raw/source path is first
    # constrained lexically, then remains unopened until mkdir(O_EXCL semantics)
    # consumes the role-global namespace below.
    manifest, artifacts = _load_authority_manifest_v5(context)
    if manifest["lifecycle"] != lifecycle:
        raise PermissionError("authority manifest lifecycle changed")
    gate = manifest["gate"]
    namespace_sha256 = _registry_namespace_v5(
        gate=gate,
        dataset_manifest_file_sha256=artifacts["dataset_manifest"][
            "file_sha256"
        ],
        role_commitment_file_sha256=artifacts[f"{gate}_role_commitment"][
            "file_sha256"
        ],
        protocol_generation=manifest["protocol_generation"],
    )
    reservation = _reservation_payload_v5(
        gate=gate,
        namespace_sha256=namespace_sha256,
        context=context,
        manifest=manifest,
        artifacts=artifacts,
        model_state_sha256=state_sha256,
    )
    reservation_content_sha256, reservation_file_sha256 = (
        _acquire_gate_reservation_v5(
            context,
            gate=gate,
            namespace_sha256=namespace_sha256,
            reservation=reservation,
        )
    )
    encoded = _open_authorized_artifacts_v5(context, artifacts)
    inputs = _validate_authorized_gate_inputs_v5(
        manifest=manifest,
        artifacts=artifacts,
        encoded=encoded,
        model_state_sha256=state_sha256,
    )
    if lifecycle == LIFECYCLE_PROMOTED:
        prior_context, prior_inputs = _load_prior_g2_inputs_v5(
            context=context,
            current_inputs=inputs,
            model_state_sha256=state_sha256,
        )
        prior_g2 = inputs.parsed["prior_g2_report"]
        prior_finalized_spec = artifacts["prior_g2_finalized"]
        g2_report = _validate_secure_gate_report_v5(
            prior_g2,
            gate="g2",
            model=model,
            inputs=prior_inputs,
            context=prior_context,
            report_path=artifacts["prior_g2_report"]["path"],
            report_file_sha256=artifacts["prior_g2_report"]["file_sha256"],
            finalized_file_sha256=prior_finalized_spec["file_sha256"],
            expected_authority_manifest_file_sha256=artifacts[
                "prior_g2_authority_manifest"
            ]["file_sha256"],
        )
        g2_report_path = artifacts["prior_g2_report"]["path"]
        g2_report_file_sha256 = artifacts["prior_g2_report"]["file_sha256"]
        g2_finalized_file_sha256 = prior_finalized_spec["file_sha256"]
    else:
        g2_report = None
        g2_report_path = None
        g2_report_file_sha256 = None
        g2_finalized_file_sha256 = None
    finalizer = (
        _finalize_g2_raw_result_v5
        if gate == "g2"
        else _finalize_g3_raw_result_v5
    )
    current_report = finalizer(
        gate=gate,
        model=model,
        inputs=inputs,
        namespace_sha256=namespace_sha256,
        reservation_content_sha256=reservation_content_sha256,
        reservation_file_sha256=reservation_file_sha256,
        authority_manifest_file_sha256=context.authority_manifest_file_sha256,
    )
    current_path, current_file_sha256, current_finalized_sha256 = (
        _write_gate_finalization_v5(
            context,
            gate=gate,
            namespace_sha256=namespace_sha256,
            report=current_report,
            reservation_content_sha256=reservation_content_sha256,
            reservation_file_sha256=reservation_file_sha256,
        )
    )
    if gate == "g2":
        g2_report = current_report
        g2_report_path = current_path
        g2_report_file_sha256 = current_file_sha256
        g2_finalized_file_sha256 = current_finalized_sha256
        g3_report = None
        g3_report_path = None
        g3_report_file_sha256 = None
        g3_finalized_file_sha256 = None
    else:
        g3_report = current_report
        g3_report_path = current_path
        g3_report_file_sha256 = current_file_sha256
        g3_finalized_file_sha256 = current_finalized_sha256
    output = shared_output_contract_v5(model)
    v4_geometry = canonical_v4_geometry_contract_v5(model.model_config)
    bev_geometry = canonical_bev_geometry_contract_v5(model.model_config)
    architecture = shared_architecture_contract_v5(model)
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_V5_SCHEMA,
        "lifecycle": lifecycle,
        "model_family": MODEL_FAMILY,
        "runtime_ready": lifecycle == LIFECYCLE_PROMOTED,
        "model_state_sha256": state_sha256,
        "model_state_dict": state,
        "model_config": model.model_config.to_dict(),
        "model_config_sha256": model.model_config.content_sha256,
        "architecture_contract": architecture,
        "architecture_contract_sha256": _canonical_sha256(architecture),
        "output_contract": output,
        "output_contract_sha256": _canonical_sha256(output),
        "v4_geometry": v4_geometry,
        "v4_geometry_sha256": _canonical_sha256(v4_geometry),
        "bev_geometry": bev_geometry,
        "bev_geometry_sha256": _canonical_sha256(bev_geometry),
        "provenance": inputs.provenance.to_dict(),
        "provenance_contract_sha256": inputs.provenance.content_sha256,
        "g2_report": g2_report,
        "g2_report_content_sha256": g2_report["content_sha256"],
        "g2_report_source": _GATE_REPORT_REGISTRY_SOURCE,
        "g2_report_file_sha256": g2_report_file_sha256,
        "g3_report": g3_report,
        "g3_report_content_sha256": (
            None if g3_report is None else g3_report["content_sha256"]
        ),
        "g3_report_source": (
            None if g3_report is None else _GATE_REPORT_REGISTRY_SOURCE
        ),
        "g3_report_file_sha256": g3_report_file_sha256,
        "authority_manifest_path": context.authority_manifest_path,
        "authority_manifest_file_sha256": (
            context.authority_manifest_file_sha256
        ),
        "authority_manifest_content_sha256": manifest["content_sha256"],
        "registry_namespace_sha256": namespace_sha256,
        "g2_report_path": g2_report_path,
        "g2_registry_finalized_file_sha256": g2_finalized_file_sha256,
        "g3_report_path": g3_report_path,
        "g3_registry_finalized_file_sha256": g3_finalized_file_sha256,
    }
    validate_checkpoint_v5_payload(
        payload,
        expected_lifecycle=lifecycle,
        context=context,
    )
    checkpoint_v5_weights_only_roundtrip(
        payload,
        expected_lifecycle=lifecycle,
        context=context,
    )
    return payload


# ---------------------------------------------------------------------------
# Canonical repository production boundary (V6 authority closure).
#
# These definitions deliberately replace the legacy filesystem candidate above.
# The legacy helpers remain only so old structure-only tests can parse historical
# payloads; public production entry points below accept no caller-created roots.

_CANONICAL_AUTHORITY_SCHEMA_V5 = PRODUCTION_AUTHORITY_MANIFEST_V5_SCHEMA
_CANONICAL_PRODUCTION_PAYLOAD_FIELDS_V5 = {
    "schema",
    "lifecycle",
    "runtime_ready",
    "model_family",
    "model_config",
    "model_config_sha256",
    "model_state_dict",
    "model_state_sha256",
    "architecture_contract",
    "architecture_contract_sha256",
    "output_contract",
    "output_contract_sha256",
    "v4_geometry",
    "v4_geometry_sha256",
    "bev_geometry",
    "bev_geometry_sha256",
    "authority_manifest_file_sha256",
    "authority_manifest_content_sha256",
    "registry_namespace_sha256",
    "reservation_content_sha256",
    "reservation_file_sha256",
    "g2_final_report",
    "g3_final_report",
}


def _make_production_context_api_v5():
    @dataclass(
        frozen=True,
        init=False,
        slots=True,
    )
    class ProductionCheckpointContextV5:
        """Legacy tombstone; production contexts no longer exist in-process."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise PermissionError(
                "production checkpoint contexts were removed; use the one-shot CLI"
            )

    def require_context(context: object) -> ProductionCheckpointContextV5:
        raise PermissionError("production checkpoint contexts were removed; use the one-shot CLI")

    def load_context() -> ProductionCheckpointContextV5:
        raise PermissionError("production checkpoint contexts were removed; use the one-shot CLI")

    return ProductionCheckpointContextV5, load_context, require_context


(
    _RemovedProductionCheckpointContextV5,
    _removed_load_production_checkpoint_context_v5,
    _require_canonical_context_v5,
) = _make_production_context_api_v5()


def _canonical_authority_artifact_v5(
    value: object,
    *,
    name: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "file_sha256"}:
        raise ValueError(f"canonical authority artifact {name} fields changed")
    path = _canonical_relative_path_v5(value.get("path"), name=f"{name} path")
    file_sha256 = value.get("file_sha256")
    _require_sha256(file_sha256, name=f"{name} file hash")
    return {"path": path, "file_sha256": str(file_sha256)}


def _load_repository_authority_v5(
    context: ProductionCheckpointContextV5,
) -> dict[str, Any]:
    context = _require_canonical_context_v5(context)
    encoded = _read_relative_file_v5(
        context.repository_root,
        context.authority_manifest_path,
        expected_file_sha256=context.authority_manifest_file_sha256,
        name="canonical Shared JEPA V5 authority",
    )
    authority = _parse_canonical_json_file_v5(
        encoded,
        name="canonical Shared JEPA V5 authority",
    )
    if set(authority) != {
        "schema",
        "protocol_generation",
        "evaluated_model_state_sha256",
        "artifacts",
        "scene_results",
        "content_sha256",
    } or authority.get("schema") != _CANONICAL_AUTHORITY_SCHEMA_V5:
        raise ValueError("canonical Shared JEPA V5 authority fields changed")
    generation = authority.get("protocol_generation")
    _require_sha256(
        authority.get("evaluated_model_state_sha256"),
        name="authority evaluated model state",
    )
    if type(generation) is not str or not generation:
        raise ValueError("canonical authority protocol generation changed")
    expected_artifacts = {
        "dataset_role_manifest",
        "evaluated_checkpoint",
        "authority_source",
        "implementation_source",
        "registry_policy_source",
        "runner_source",
        "finalizer_core_source",
        "g2_finalizer_source",
        "g3_finalizer_source",
        "g2_role_commitment",
        "g3_role_commitment",
        "g2_runner_ledger",
        "g3_runner_ledger",
        "g2_final_report",
        "g3_final_report",
    }
    raw_artifacts = authority.get("artifacts")
    if not isinstance(raw_artifacts, Mapping) or set(raw_artifacts) != expected_artifacts:
        raise ValueError("canonical authority artifact inventory changed")
    authority["artifacts"] = {
        name: _canonical_authority_artifact_v5(raw_artifacts[name], name=name)
        for name in sorted(expected_artifacts)
    }
    scene_results = authority.get("scene_results")
    if not isinstance(scene_results, Mapping) or set(scene_results) != {"g2", "g3"}:
        raise ValueError("canonical scene-result inventory changed")
    normalized_results: dict[str, dict[str, dict[str, str]]] = {}
    for gate in ("g2", "g3"):
        rows = scene_results[gate]
        if not isinstance(rows, Mapping) or not rows:
            raise ValueError(f"canonical {gate} scene-result inventory is empty")
        normalized_results[gate] = {
            str(scene_id): _canonical_authority_artifact_v5(
                spec,
                name=f"{gate} scene result {scene_id}",
            )
            for scene_id, spec in sorted(rows.items())
        }
    authority["scene_results"] = normalized_results
    from lewm.models.shared_observable_camera_ray_jepa_v5_authority import (
        require_frozen_production_authority,
    )

    frozen = require_frozen_production_authority()
    artifacts = authority["artifacts"]
    for artifact_name, frozen_name in (
        ("dataset_role_manifest", "dataset_role_manifest_file_sha256"),
        ("g2_runner_ledger", "g2_runner_ledger_file_sha256"),
        ("g3_runner_ledger", "g3_runner_ledger_file_sha256"),
        ("g2_final_report", "g2_final_report_file_sha256"),
        ("g3_final_report", "g3_final_report_file_sha256"),
    ):
        if artifacts[artifact_name]["file_sha256"] != frozen[frozen_name]:
            raise PermissionError(f"canonical {artifact_name} hash is not frozen")
    return authority


def _open_repository_json_v5(
    context: ProductionCheckpointContextV5,
    spec: Mapping[str, str],
    *,
    name: str,
) -> dict[str, Any]:
    encoded = _read_relative_file_v5(
        context.repository_root,
        spec["path"],
        expected_file_sha256=spec["file_sha256"],
        name=name,
    )
    return _parse_canonical_json_file_v5(encoded, name=name)


def _validate_distinct_authorized_sources_v5(
    context: ProductionCheckpointContextV5,
    authority: Mapping[str, Any],
) -> str:
    from lewm.benchmarks import (
        finalize_shared_observable_camera_ray_jepa_v5_g2 as g2_finalizer,
    )
    from lewm.benchmarks import (
        finalize_shared_observable_camera_ray_jepa_v5_g3 as g3_finalizer,
    )
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_runner_policy
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_finalizer_core
    from lewm.models import shared_observable_camera_ray_jepa_v5_authority
    from lewm.models import shared_observable_camera_ray_jepa_v5_registry_policy

    artifacts = authority["artifacts"]
    expected_sources = {
        "authority_source": Path(
            shared_observable_camera_ray_jepa_v5_authority.__file__
        ),
        "implementation_source": Path(__file__),
        "registry_policy_source": Path(
            shared_observable_camera_ray_jepa_v5_registry_policy.__file__
        ),
        "runner_source": Path(
            shared_observable_camera_ray_jepa_v5_runner_policy.__file__
        ),
        "finalizer_core_source": Path(
            shared_observable_camera_ray_jepa_v5_finalizer_core.__file__
        ),
        "g2_finalizer_source": Path(g2_finalizer.__file__),
        "g3_finalizer_source": Path(g3_finalizer.__file__),
    }
    hashes: dict[str, str] = {}
    for name, source_path in expected_sources.items():
        spec = artifacts[name]
        expected_path = (context.repository_root / spec["path"]).resolve(strict=True)
        if source_path.resolve(strict=True) != expected_path:
            raise PermissionError(f"authorized {name} was imported from another root")
        encoded = _read_relative_file_v5(
            context.repository_root,
            spec["path"],
            expected_file_sha256=spec["file_sha256"],
            name=f"authorized {name}",
        )
        current = source_path.read_bytes()
        if encoded != current:
            raise PermissionError(f"authorized {name} differs from reviewed source")
        hashes[name] = spec["file_sha256"]
    if len(set(hashes.values())) != len(hashes):
        raise PermissionError("authority, implementation, registry, runner, and finalizers overlap")
    return hashes["runner_source"]


def _recompute_canonical_gate_report_v5(
    *,
    gate: str,
    context: ProductionCheckpointContextV5,
    authority: Mapping[str, Any],
    model_state_sha256: str,
) -> dict[str, Any]:
    from lewm.benchmarks.finalize_shared_observable_camera_ray_jepa_v5_g2 import (
        finalize_g2,
    )
    from lewm.benchmarks.finalize_shared_observable_camera_ray_jepa_v5_g3 import (
        finalize_g3,
    )
    from lewm.benchmarks.shared_observable_camera_ray_jepa_v5_runner_policy import (
        reopen_canonical_runner_batch,
    )

    artifacts = authority["artifacts"]
    runner_source_sha256 = _validate_distinct_authorized_sources_v5(
        context,
        authority,
    )
    finalizer = finalize_g2 if gate == "g2" else finalize_g3
    runner_batch = reopen_canonical_runner_batch(
        gate=gate,
        expected_model_state_sha256=model_state_sha256,
    )
    recomputed = finalizer(
        runner_batch=runner_batch,
        expected_model_state_sha256=model_state_sha256,
        expected_checkpoint_file_sha256=artifacts["evaluated_checkpoint"][
            "file_sha256"
        ],
        expected_runner_source_sha256=runner_source_sha256,
    )
    finalized = _open_repository_json_v5(
        context,
        artifacts[f"{gate}_final_report"],
        name=f"canonical independently finalized {gate} report",
    )
    if (
        finalized != recomputed
        or finalized.get("passed") is not True
        or finalized.get("production_authority_eligible") is not True
        or finalized.get("synthetic_only") is not False
    ):
        raise PermissionError(f"canonical {gate} final report did not recompute")
    return finalized


def _canonical_production_payload_v5(
    *,
    model: SharedObservableCameraRayJepaV5,
    lifecycle: str,
    context: ProductionCheckpointContextV5,
    authority: Mapping[str, Any],
    namespace_sha256: str,
    reservation_content_sha256: str,
    reservation_file_sha256: str,
    g2_report: Mapping[str, Any],
    g3_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    state = model.deployment_state_dict()
    state_sha256 = tensor_state_dict_sha256(state)
    architecture = shared_architecture_contract_v5(model)
    output = shared_output_contract_v5(model)
    v4_geometry = canonical_v4_geometry_contract_v5(model.model_config)
    bev_geometry = canonical_bev_geometry_contract_v5(model.model_config)
    return {
        "schema": CHECKPOINT_V5_SCHEMA,
        "lifecycle": lifecycle,
        "runtime_ready": lifecycle == LIFECYCLE_PROMOTED,
        "model_family": MODEL_FAMILY,
        "model_config": model.model_config.to_dict(),
        "model_config_sha256": model.model_config.content_sha256,
        "model_state_dict": state,
        "model_state_sha256": state_sha256,
        "architecture_contract": architecture,
        "architecture_contract_sha256": _canonical_sha256(architecture),
        "output_contract": output,
        "output_contract_sha256": _canonical_sha256(output),
        "v4_geometry": v4_geometry,
        "v4_geometry_sha256": _canonical_sha256(v4_geometry),
        "bev_geometry": bev_geometry,
        "bev_geometry_sha256": _canonical_sha256(bev_geometry),
        "authority_manifest_file_sha256": context.authority_manifest_file_sha256,
        "authority_manifest_content_sha256": authority["content_sha256"],
        "registry_namespace_sha256": namespace_sha256,
        "reservation_content_sha256": reservation_content_sha256,
        "reservation_file_sha256": reservation_file_sha256,
        "g2_final_report": dict(g2_report),
        "g3_final_report": None if g3_report is None else dict(g3_report),
    }


def _validate_canonical_reservation_v5(
    *,
    context: ProductionCheckpointContextV5,
    authority: Mapping[str, Any],
    lifecycle: str,
    model_state_sha256: str,
    namespace_sha256: object,
    reservation_content_sha256: object,
    reservation_file_sha256: object,
) -> None:
    from lewm.models.shared_observable_camera_ray_jepa_v5_registry_policy import (
        role_global_namespace,
    )

    gate = "g2" if lifecycle == LIFECYCLE_G3_CANDIDATE else "g3"
    artifacts = authority["artifacts"]
    expected_namespace = role_global_namespace(
        gate=gate,
        dataset_role_manifest_file_sha256=artifacts["dataset_role_manifest"][
            "file_sha256"
        ],
        role_commitment_file_sha256=artifacts[f"{gate}_role_commitment"][
            "file_sha256"
        ],
        protocol_generation=authority["protocol_generation"],
    )
    if namespace_sha256 != expected_namespace:
        raise PermissionError("canonical checkpoint registry namespace changed")
    _require_sha256(
        reservation_content_sha256,
        name="canonical reservation content hash",
    )
    _require_sha256(
        reservation_file_sha256,
        name="canonical reservation file hash",
    )
    encoded = _read_relative_file_v5(
        context.registry_root,
        f"{gate}/{expected_namespace}/reservation.json",
        expected_file_sha256=str(reservation_file_sha256),
        name=f"canonical {gate} reservation",
    )
    reservation = _parse_canonical_json_file_v5(
        encoded,
        name=f"canonical {gate} reservation",
    )
    if set(reservation) != {
        "schema",
        "authority_manifest_file_sha256",
        "model_state_sha256",
        "namespace_sha256",
        "gate",
        "content_sha256",
    } or (
        reservation.get("schema")
        != f"lewm_go2_shared_jepa_{gate}_canonical_reservation_v6"
        or reservation.get("authority_manifest_file_sha256")
        != context.authority_manifest_file_sha256
        or reservation.get("model_state_sha256") != model_state_sha256
        or reservation.get("namespace_sha256") != expected_namespace
        or reservation.get("gate") != gate
        or reservation.get("content_sha256") != reservation_content_sha256
    ):
        raise PermissionError("canonical checkpoint reservation changed")


def _removed_build_checkpoint_v5_payload(
    model: SharedObservableCameraRayJepaV5,
    *,
    lifecycle: str,
    context: ProductionCheckpointContextV5,
) -> dict[str, Any]:
    """Removed: publication is an isolated fixed-file CLI stage."""

    raise PermissionError("production checkpoint construction was removed; use the one-shot publisher CLI")


def _removed_in_process_build_checkpoint_v5_payload_v5(
    model: SharedObservableCameraRayJepaV5,
    *,
    lifecycle: str,
    context: ProductionCheckpointContextV5,
) -> dict[str, Any]:
    """Unreachable historical implementation retained for source archaeology."""

    from lewm.models.shared_observable_camera_ray_jepa_v5_registry_policy import (
        acquire_canonical_attempt,
        role_global_namespace,
    )

    if not isinstance(model, SharedObservableCameraRayJepaV5):
        raise TypeError("model must be SharedObservableCameraRayJepaV5")
    context = _require_canonical_context_v5(context)
    if lifecycle not in {LIFECYCLE_G3_CANDIDATE, LIFECYCLE_PROMOTED}:
        raise ValueError("checkpoint lifecycle must be g3_candidate or promoted")
    _require_production_checkpoint_config(model.model_config)
    model_state_sha256 = tensor_state_dict_sha256(model.deployment_state_dict())

    # The fixed authority is the only file opened before role-global reservation.
    authority = _load_repository_authority_v5(context)
    if authority["evaluated_model_state_sha256"] != model_state_sha256:
        raise PermissionError("model state differs from frozen evaluation authority")
    gate = "g2" if lifecycle == LIFECYCLE_G3_CANDIDATE else "g3"
    artifacts = authority["artifacts"]
    namespace = role_global_namespace(
        gate=gate,
        dataset_role_manifest_file_sha256=artifacts["dataset_role_manifest"][
            "file_sha256"
        ],
        role_commitment_file_sha256=artifacts[f"{gate}_role_commitment"][
            "file_sha256"
        ],
        protocol_generation=authority["protocol_generation"],
    )
    reservation_core = {
        "schema": f"lewm_go2_shared_jepa_{gate}_canonical_reservation_v6",
        "authority_manifest_file_sha256": context.authority_manifest_file_sha256,
        "model_state_sha256": model_state_sha256,
    }
    reservation_content_sha256, reservation_file_sha256 = (
        acquire_canonical_attempt(
            gate=gate,
            namespace_sha256=namespace,
            reservation=reservation_core,
        )
    )
    if lifecycle == LIFECYCLE_PROMOTED:
        prior_authority = _load_repository_authority_v5(context)
        if prior_authority != authority:
            raise PermissionError("prior G2 authority changed during G3 promotion")
    g2 = _recompute_canonical_gate_report_v5(
        gate="g2",
        context=context,
        authority=authority,
        model_state_sha256=model_state_sha256,
    )
    g3 = (
        _recompute_canonical_gate_report_v5(
            gate="g3",
            context=context,
            authority=authority,
            model_state_sha256=model_state_sha256,
        )
        if lifecycle == LIFECYCLE_PROMOTED
        else None
    )
    payload = _canonical_production_payload_v5(
        model=model,
        lifecycle=lifecycle,
        context=context,
        authority=authority,
        namespace_sha256=namespace,
        reservation_content_sha256=reservation_content_sha256,
        reservation_file_sha256=reservation_file_sha256,
        g2_report=g2,
        g3_report=g3,
    )
    validate_checkpoint_v5_payload(
        payload,
        expected_lifecycle=lifecycle,
        context=context,
    )
    return payload


def _removed_validate_checkpoint_v5_payload(
    payload: Mapping[str, Any],
    *,
    expected_lifecycle: str | None = None,
    context: ProductionCheckpointContextV5 | None = None,
) -> None:
    raise PermissionError("production checkpoint validation was removed; use the one-shot publisher CLI")


def _removed_in_process_validate_checkpoint_v5_payload_v5(
    payload: Mapping[str, Any],
    *,
    expected_lifecycle: str | None = None,
    context: ProductionCheckpointContextV5 | None = None,
) -> None:
    """Unreachable historical implementation retained for source archaeology."""

    context = _require_canonical_context_v5(context)
    if not isinstance(payload, Mapping) or set(payload) != (
        _CANONICAL_PRODUCTION_PAYLOAD_FIELDS_V5
    ):
        raise ValueError("canonical production checkpoint fields changed")
    lifecycle = payload.get("lifecycle")
    if expected_lifecycle is not None and lifecycle != expected_lifecycle:
        raise PermissionError("canonical checkpoint lifecycle changed")
    if lifecycle not in {LIFECYCLE_G3_CANDIDATE, LIFECYCLE_PROMOTED}:
        raise ValueError("canonical checkpoint lifecycle is invalid")
    if (
        payload.get("schema") != CHECKPOINT_V5_SCHEMA
        or payload.get("model_family") != MODEL_FAMILY
        or payload.get("runtime_ready") is not (lifecycle == LIFECYCLE_PROMOTED)
    ):
        raise PermissionError("canonical checkpoint identity changed")
    config = SharedObservableCameraRayJepaV5Config.from_mapping(
        _require_mapping(payload.get("model_config"), "model_config")
    )
    _require_production_checkpoint_config(config)
    if payload.get("model_config_sha256") != config.content_sha256:
        raise ValueError("canonical checkpoint config hash changed")
    model = SharedObservableCameraRayJepaV5(config)
    expected_contracts = {
        "architecture_contract": shared_architecture_contract_v5(model),
        "output_contract": shared_output_contract_v5(model),
        "v4_geometry": canonical_v4_geometry_contract_v5(config),
        "bev_geometry": canonical_bev_geometry_contract_v5(config),
    }
    for name, expected in expected_contracts.items():
        if (
            payload.get(name) != expected
            or payload.get(f"{name}_sha256") != _canonical_sha256(expected)
        ):
            raise ValueError(f"canonical checkpoint {name} changed")
    state = _require_mapping(payload.get("model_state_dict"), "model_state_dict")
    _validate_state_contract(state, model.deployment_state_dict())
    state_sha256 = tensor_state_dict_sha256(state)
    if payload.get("model_state_sha256") != state_sha256:
        raise ValueError("canonical checkpoint model-state hash changed")
    model.load_deployment_state_dict(state)
    authority = _load_repository_authority_v5(context)
    if (
        payload.get("authority_manifest_file_sha256")
        != context.authority_manifest_file_sha256
        or payload.get("authority_manifest_content_sha256")
        != authority["content_sha256"]
        or authority["evaluated_model_state_sha256"] != state_sha256
    ):
        raise PermissionError("canonical checkpoint authority changed")
    _validate_canonical_reservation_v5(
        context=context,
        authority=authority,
        lifecycle=str(lifecycle),
        model_state_sha256=state_sha256,
        namespace_sha256=payload.get("registry_namespace_sha256"),
        reservation_content_sha256=payload.get("reservation_content_sha256"),
        reservation_file_sha256=payload.get("reservation_file_sha256"),
    )
    g2 = _recompute_canonical_gate_report_v5(
        gate="g2",
        context=context,
        authority=authority,
        model_state_sha256=state_sha256,
    )
    g3 = (
        _recompute_canonical_gate_report_v5(
            gate="g3",
            context=context,
            authority=authority,
            model_state_sha256=state_sha256,
        )
        if lifecycle == LIFECYCLE_PROMOTED
        else None
    )
    if payload.get("g2_final_report") != g2 or payload.get("g3_final_report") != g3:
        raise PermissionError("canonical checkpoint finalized evidence changed")


def _removed_checkpoint_v5_weights_only_roundtrip(
    payload: Mapping[str, Any],
    *,
    expected_lifecycle: str | None = None,
    context: ProductionCheckpointContextV5 | None = None,
) -> dict[str, Any]:
    context = _require_canonical_context_v5(context)
    validate_checkpoint_v5_payload(
        payload,
        expected_lifecycle=expected_lifecycle,
        context=context,
    )
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)
    if not isinstance(restored, dict):
        raise ValueError("canonical checkpoint roundtrip changed root type")
    validate_checkpoint_v5_payload(
        restored,
        expected_lifecycle=expected_lifecycle,
        context=context,
    )
    return restored


__all__ = [
    "CHECKPOINT_V5_SCHEMA",
    "CheckpointProvenanceV5",
    "EstablishedJepaPackageV5",
    "G2_GATE_REPORT_V5_SCHEMA",
    "G2_GATE_METRICS_V5",
    "G3_GATE_REPORT_V5_SCHEMA",
    "G3_GATE_METRICS_V5",
    "JepaCounterfactualsV5",
    "LIFECYCLE_G3_CANDIDATE",
    "LIFECYCLE_PROMOTED",
    "MODEL_FAMILY",
    "ObservableCameraRayEvidenceV4Head",
    "PRODUCTION_AUTHORITY_MANIFEST_V5_SCHEMA",
    "PRODUCTION_MODEL_CONFIG_V5_SCHEMA",
    "ObservableCameraRayV4FrameSupervisionV5",
    "SharedHierarchicalV4LossV5",
    "SharedJointLossV5",
    "SharedObservableCameraRayV4FrameLossV5",
    "SharedObservableCameraRayV4LossV5",
    "SharedObservableCameraRayJepaV5",
    "SharedObservableCameraRayJepaV5Config",
    "SharedOnlineFrameV5",
    "SharedTrainingPairV5",
    "SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA",
    "V4HeadMigrationReceiptV5",
    "canonical_bev_geometry_contract_v5",
    "canonical_v4_geometry_contract_v5",
    "checkpoint_contract_bindings_v5",
    "gate_thresholds_v5",
    "shared_architecture_contract_v5",
    "shared_output_contract_v5",
    "tensor_state_dict_sha256",
]
