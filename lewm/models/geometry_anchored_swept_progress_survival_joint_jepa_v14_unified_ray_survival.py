"""Unified ray-survival Camera bottleneck for the joint JEPA V14.

V14 keeps the V13 RGB encoder, first-hit hazards, within-bin offsets, retained
evidence planes, and joint-JEPA state shapes.  Its sole scientific change is
to remove the independent ground-clear MLP: clear-to-ground evidence is the
fractional survival probability of the same ordered hazard ray that supplies
occupied first-hit evidence.
"""
from __future__ import annotations

import copy
import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    PIXEL_RAY_SHAPE,
    SOURCE_SHAPE,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    DEFAULT_QUERY_CHUNK_SIZE,
    DENSE_FEATURE_DIM,
    DEPTH_BIN_COUNT,
    DEPTH_BIN_SIZE_M,
    DEPTH_NEAR_EDGE_M,
    ENCODER_DIM,
    PATCH_SIZE,
    GroundQueryGeometryV4,
    ObservableCameraRayEvidenceV4Model,
)
from lewm.models.shared_observable_camera_ray_jepa_v5 import (
    ObservableCameraRayEvidenceV4Head,
)

from .geometry_anchored_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition import (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredSweptProgressSurvivalJointJepaV12,
)
from .geometry_anchored_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck import (
    CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13,
    PREDICTOR_GROUP_PARAMETER_COUNT_V13,
    PREDICTOR_PARAMETER_PREFIXES_V13,
    PROJECTION_INITIALIZATION_SEED_V13 as PROJECTION_INITIALIZATION_SEED_V14,
    REPRESENTATION_GROUP_PARAMETER_COUNT_V13,
    REPRESENTATION_PARAMETER_PREFIXES_V13,
    SHARED_PARAMETER_PREFIXES_V13,
    TARGET_PARAMETER_PREFIXES_V13,
    CameraEvidenceBottleneckAuxiliaryEncodingV13,
    CameraEvidenceBottleneckEncodingV13,
    CameraEvidenceBottleneckLiftV13,
    GeometryAnchoredSweptProgressSurvivalJointJepaV13,
)


REMOVED_GROUND_HEAD_PARAMETER_COUNT_V14 = 2_689
SHARED_ROUTE_PARAMETER_COUNT_V14 = 3_102_824
REPRESENTATION_GROUP_PARAMETER_COUNT_V14 = REPRESENTATION_GROUP_PARAMETER_COUNT_V13
PREDICTOR_GROUP_PARAMETER_COUNT_V14 = PREDICTOR_GROUP_PARAMETER_COUNT_V13
ONLINE_TRAINABLE_PARAMETER_COUNT_V14 = 3_383_917
TARGET_BOTTLENECK_PARAMETER_COUNT_V14 = 3_106_216
MIGRATED_EVIDENCE_STATE_COUNT_V14 = 11
# Compatibility spellings used by the unchanged V13 training/executor hooks.
SHARED_ROUTE_PARAMETER_COUNT_V13 = SHARED_ROUTE_PARAMETER_COUNT_V14
ONLINE_TRAINABLE_PARAMETER_COUNT_V13 = ONLINE_TRAINABLE_PARAMETER_COUNT_V14
TARGET_BOTTLENECK_PARAMETER_COUNT_V13 = TARGET_BOTTLENECK_PARAMETER_COUNT_V14
PROJECTION_INITIALIZATION_SEED_V13 = PROJECTION_INITIALIZATION_SEED_V14


def sample_hazards_at_ground_queries_v14(
    hazard_logits: torch.Tensor,
    query_geometry: GroundQueryGeometryV4,
    *,
    query_chunk_size: int | None,
) -> torch.Tensor:
    """Sample all hazard depths with exact border-preserving ray geometry."""

    if (
        not isinstance(hazard_logits, torch.Tensor)
        or hazard_logits.ndim != 4
        or hazard_logits.shape[1] != DEPTH_BIN_COUNT
        or not hazard_logits.is_floating_point()
    ):
        raise ValueError("hazard_logits must have floating shape (B,64,Hray,Wray)")
    if not isinstance(query_geometry, GroundQueryGeometryV4):
        raise TypeError("query_geometry must be GroundQueryGeometryV4")
    if hazard_logits.shape[0] != query_geometry.in_frustum.shape[0]:
        raise ValueError("hazard and ground-query batches differ")
    if hazard_logits.device != query_geometry.sample_grid.device:
        raise ValueError("hazards and ground queries must share a device")

    batch = hazard_logits.shape[0]
    query_shape = tuple(query_geometry.in_frustum.shape[1:])
    flat_grid = query_geometry.sample_grid.reshape(batch, -1, 2)
    query_count = flat_grid.shape[1]
    if query_chunk_size is None:
        chunk_size = query_count
    else:
        chunk_size = int(query_chunk_size)
        if chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive or None")
    sampled_chunks = []
    for start in range(0, query_count, chunk_size):
        grid = flat_grid[:, start : start + chunk_size, None, :].to(
            dtype=hazard_logits.dtype
        )
        sampled = F.grid_sample(
            hazard_logits,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled_chunks.append(sampled.squeeze(-1).transpose(1, 2))
    sampled_flat = torch.cat(sampled_chunks, dim=1)
    return sampled_flat.reshape(batch, *query_shape, DEPTH_BIN_COUNT)


def fractional_ray_log_survival_v14(
    hazard_logits: torch.Tensor,
    query_geometry: GroundQueryGeometryV4,
    *,
    query_chunk_size: int | None,
) -> torch.Tensor:
    """Return log survival from the near edge to each query distance.

    Each ordered 10 cm hazard bin is treated as a constant integrated hazard.
    A query partway through a bin therefore contributes the corresponding
    fraction of that bin's log-survival.  Distances before the near edge clamp
    to certain survival and distances beyond the far edge use all 64 bins.
    """

    if (
        not isinstance(hazard_logits, torch.Tensor)
        or hazard_logits.ndim != 4
        or hazard_logits.shape[1] != DEPTH_BIN_COUNT
        or not hazard_logits.is_floating_point()
    ):
        raise ValueError("hazard_logits must have floating shape (B,64,Hray,Wray)")
    if not isinstance(query_geometry, GroundQueryGeometryV4):
        raise TypeError("query_geometry must be GroundQueryGeometryV4")

    sampled_hazard = sample_hazards_at_ground_queries_v14(
        hazard_logits,
        query_geometry,
        query_chunk_size=query_chunk_size,
    )
    log_survive = F.logsigmoid(-sampled_hazard)
    distance_m = query_geometry.target_distance_m.to(
        device=hazard_logits.device,
        dtype=hazard_logits.dtype,
    )
    continuous_bin = ((distance_m - DEPTH_NEAR_EDGE_M) / DEPTH_BIN_SIZE_M).clamp(
        min=0.0,
        max=float(DEPTH_BIN_COUNT),
    )
    whole_bin_count = torch.floor(continuous_bin).to(dtype=torch.long)
    fractional_bin = continuous_bin - whole_bin_count.to(continuous_bin.dtype)

    prefix = torch.cat(
        (
            torch.zeros_like(log_survive[..., :1]),
            torch.cumsum(log_survive, dim=-1),
        ),
        dim=-1,
    )
    whole_survival = torch.gather(
        prefix,
        dim=-1,
        index=whole_bin_count[..., None],
    ).squeeze(-1)
    current_survival = torch.gather(
        log_survive,
        dim=-1,
        index=whole_bin_count.clamp(max=DEPTH_BIN_COUNT - 1)[..., None],
    ).squeeze(-1)
    partial_survival = torch.where(
        whole_bin_count < DEPTH_BIN_COUNT,
        fractional_bin * current_survival,
        torch.zeros_like(current_survival),
    )
    return whole_survival + partial_survival


def finite_ground_clear_logits_v14(log_clear_probability: torch.Tensor) -> torch.Tensor:
    """Convert log clear probability to finite Bernoulli logits stably."""

    if (
        not isinstance(log_clear_probability, torch.Tensor)
        or not log_clear_probability.is_floating_point()
    ):
        raise ValueError("log_clear_probability must be a floating tensor")
    epsilon = torch.finfo(log_clear_probability.dtype).eps
    bounded_log_clear = log_clear_probability.clamp(
        max=math.log1p(-epsilon),
    )
    log_blocked = torch.log(
        (-torch.expm1(bounded_log_clear)).clamp_min(
            torch.finfo(log_clear_probability.dtype).tiny
        )
    )
    return bounded_log_clear - log_blocked


class UnifiedRaySurvivalCameraEvidenceHeadV14(ObservableCameraRayEvidenceV4Head):
    """Encoder-free V4 pixel head with no independent ground parameters."""

    def __init__(
        self,
        *,
        source_shape: tuple[int, int] = SOURCE_SHAPE,
        pixel_ray_shape: tuple[int, int] = PIXEL_RAY_SHAPE,
        query_chunk_size: int | None = DEFAULT_QUERY_CHUNK_SIZE,
    ) -> None:
        nn.Module.__init__(self)
        self.source_shape = ObservableCameraRayEvidenceV4Model._positive_shape(
            source_shape,
            name="source_shape",
        )
        self.pixel_ray_shape = ObservableCameraRayEvidenceV4Model._positive_shape(
            pixel_ray_shape,
            name="pixel_ray_shape",
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
        nn.init.xavier_uniform_(self.pixel_head.weight)
        nn.init.zeros_(self.pixel_head.bias)
        with torch.no_grad():
            self.pixel_head.bias[:DEPTH_BIN_COUNT] = -math.log(
                float(DEPTH_BIN_COUNT - 1)
            )

        support = ObservableCameraRayEvidenceV4Model._canonical_support_xy(
            self.source_shape
        )
        self.register_buffer(
            "canonical_ground_support_xy_body_m",
            support,
            persistent=True,
        )

    def ground_survival_branch(
        self,
        hazard_logits: torch.Tensor,
        query_geometry: GroundQueryGeometryV4,
        *,
        query_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Derive clear-to-target logits from the shared first-hit hazards."""

        chunk_size = (
            self.query_chunk_size
            if query_chunk_size is None
            else int(query_chunk_size)
        )
        log_clear = fractional_ray_log_survival_v14(
            hazard_logits,
            query_geometry,
            query_chunk_size=chunk_size,
        )
        logits = finite_ground_clear_logits_v14(log_clear)
        return torch.where(
            query_geometry.in_frustum,
            logits,
            torch.zeros_like(logits),
        )

    def migrate_from_fit_model(
        self,
        fit_model: ObservableCameraRayEvidenceV4Model,
    ) -> tuple[str, ...]:
        """Copy only dense/pixel/geometry state; the V4 ground MLP is retired."""

        if not isinstance(fit_model, ObservableCameraRayEvidenceV4Model):
            raise TypeError("fit_model must be an ObservableCameraRayEvidenceV4Model")
        if (
            tuple(fit_model.source_shape) != self.source_shape
            or tuple(fit_model.pixel_ray_shape) != self.pixel_ray_shape
        ):
            raise ValueError("fit-model and V14 head geometries differ")
        source_state = fit_model.state_dict()
        selected = {
            name: value
            for name, value in source_state.items()
            if name.startswith(("dense_decoder.", "pixel_head."))
            or name == "canonical_ground_support_xy_body_m"
        }
        expected = set(self.state_dict())
        if set(selected) != expected or len(selected) != MIGRATED_EVIDENCE_STATE_COUNT_V14:
            raise ValueError("V14 selected evidence migration contract changed")
        self.load_state_dict(selected, strict=True)
        return tuple(sorted(selected))


class UnifiedRaySurvivalCameraEvidenceBottleneckLiftV14(
    CameraEvidenceBottleneckLiftV13
):
    """V13 evidence-plane lift driven by one unified learned ray field."""

    def __init__(self, n320_fit_model: ObservableCameraRayEvidenceV4Model) -> None:
        super().__init__(n320_fit_model)
        caller_rng = torch.random.get_rng_state().clone()
        try:
            evidence_head = UnifiedRaySurvivalCameraEvidenceHeadV14(
                source_shape=SOURCE_SHAPE,
                pixel_ray_shape=PIXEL_RAY_SHAPE,
                query_chunk_size=DEFAULT_QUERY_CHUNK_SIZE,
            )
            migrated = evidence_head.migrate_from_fit_model(n320_fit_model)
            self.evidence_head = evidence_head
            self.migrated_evidence_state_names = migrated
        finally:
            torch.random.set_rng_state(caller_rng)

    def forward_with_evidence(
        self,
        patch_tokens: torch.Tensor,
    ) -> CameraEvidenceBottleneckEncodingV13:
        """Decode one shared hazard field into the unchanged nominal state."""

        dense = self.evidence_head.decode_dense_features(patch_tokens)
        nominal_query = self._nominal_ground_query(
            batch=patch_tokens.shape[0],
            dtype=patch_tokens.dtype,
            device=patch_tokens.device,
        )
        hazard, offset = self.evidence_head.pixel_branch(dense)
        ground = self.evidence_head.ground_survival_branch(
            hazard,
            nominal_query,
            query_chunk_size=self.evidence_head.query_chunk_size,
        )
        nominal = self._raw_evidence(hazard, offset, ground, nominal_query)
        return self._latent_from_nominal_evidence(nominal)

    def forward_with_auxiliary_evidence(
        self,
        patch_tokens: torch.Tensor,
        *,
        camera_origin_body_m: torch.Tensor,
        camera_basis_body_fru: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
    ) -> CameraEvidenceBottleneckAuxiliaryEncodingV13:
        """Reuse one hazard field for nominal and calibrated ground queries."""

        dense = self.evidence_head.decode_dense_features(patch_tokens)
        nominal_query = self._nominal_ground_query(
            batch=patch_tokens.shape[0],
            dtype=patch_tokens.dtype,
            device=patch_tokens.device,
        )
        hazard, offset = self.evidence_head.pixel_branch(dense)
        nominal_ground = self.evidence_head.ground_survival_branch(
            hazard,
            nominal_query,
            query_chunk_size=self.evidence_head.query_chunk_size,
        )
        nominal = self._raw_evidence(hazard, offset, nominal_ground, nominal_query)
        encoded = self._latent_from_nominal_evidence(nominal)

        auxiliary_query = self.evidence_head.ground_query_geometry(
            camera_origin_body_m,
            camera_basis_body_fru,
            ground_plane_z_body_m,
        )
        if auxiliary_query.in_frustum.shape[0] != patch_tokens.shape[0]:
            raise ValueError("patch-token and auxiliary calibration batches differ")
        auxiliary_ground = self.evidence_head.ground_survival_branch(
            hazard,
            auxiliary_query,
            query_chunk_size=self.evidence_head.query_chunk_size,
        )
        auxiliary = self._raw_evidence(
            hazard,
            offset,
            auxiliary_ground,
            auxiliary_query,
        )
        return CameraEvidenceBottleneckAuxiliaryEncodingV13(
            latent=encoded.latent,
            nominal_evidence=encoded.nominal_evidence,
            auxiliary_evidence=auxiliary,
            free_evidence_planes=encoded.free_evidence_planes,
            occupied_evidence_planes=encoded.occupied_evidence_planes,
        )


class GeometryAnchoredSweptProgressSurvivalJointJepaV14(
    GeometryAnchoredSweptProgressSurvivalJointJepaV13
):
    """V13 joint JEPA with unified learned FREE/OCCUPIED ray survival."""

    def __init__(
        self,
        n320_fit_model: ObservableCameraRayEvidenceV4Model,
        sweep_masks: torch.Tensor,
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        if not isinstance(n320_fit_model, ObservableCameraRayEvidenceV4Model):
            raise TypeError("n320_fit_model must be ObservableCameraRayEvidenceV4Model")
        n320_encoder_state_dict: Mapping[str, torch.Tensor] = {
            name: value.detach()
            for name, value in n320_fit_model.encoder.state_dict().items()
        }
        GeometryAnchoredSweptProgressSurvivalJointJepaV12.__init__(
            self,
            n320_encoder_state_dict,
            sweep_masks,
            config,
        )
        if int(self.target_hard_sync_count.item()) != 1:
            raise RuntimeError("predecessor construction hard-sync count changed")
        if int(self.ema_update_count.item()) != 0:
            raise RuntimeError("predecessor construction EMA count changed")

        self.bev_lift = UnifiedRaySurvivalCameraEvidenceBottleneckLiftV14(
            n320_fit_model
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_bev_lift = copy.deepcopy(self.bev_lift)
        self.target_hard_sync_count.fill_(1)
        self.ema_update_count.zero_()
        self._freeze_target()
        self._assert_final_target_identity()
        self._assert_parameter_accounting()

    def trainable_parameter_groups_v14(self):
        """Return the unchanged three V13 route groups under the V14 name."""

        return self.trainable_parameter_groups_v13()

    def _assert_parameter_accounting(self) -> None:
        groups = self.trainable_parameter_groups_v14()
        counts = tuple(
            sum(parameter.numel() for _, parameter in group) for group in groups
        )
        expected = (
            SHARED_ROUTE_PARAMETER_COUNT_V14,
            REPRESENTATION_GROUP_PARAMETER_COUNT_V14,
            PREDICTOR_GROUP_PARAMETER_COUNT_V14,
        )
        if counts != expected:
            raise RuntimeError(f"V14 online parameter-group counts changed: {counts}")
        if sum(counts) != ONLINE_TRAINABLE_PARAMETER_COUNT_V14:
            raise RuntimeError("V14 total online trainable parameter count changed")
        target_count = sum(
            parameter.numel()
            for module in self.target_modules()
            for parameter in module.parameters()
        )
        if target_count != TARGET_BOTTLENECK_PARAMETER_COUNT_V14:
            raise RuntimeError("V14 target bottleneck parameter count changed")


CameraEvidenceBottleneckLiftV14 = (
    UnifiedRaySurvivalCameraEvidenceBottleneckLiftV14
)
CameraEvidenceBottleneckJointJepaV14 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV14
)
GeometryAnchoredSweptProgressSurvivalJointJepaV14Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)


__all__ = [
    "CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13",
    "CameraEvidenceBottleneckJointJepaV14",
    "CameraEvidenceBottleneckLiftV14",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV14",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV14Config",
    "MIGRATED_EVIDENCE_STATE_COUNT_V14",
    "ONLINE_TRAINABLE_PARAMETER_COUNT_V13",
    "ONLINE_TRAINABLE_PARAMETER_COUNT_V14",
    "PREDICTOR_GROUP_PARAMETER_COUNT_V13",
    "PREDICTOR_GROUP_PARAMETER_COUNT_V14",
    "PREDICTOR_PARAMETER_PREFIXES_V13",
    "PROJECTION_INITIALIZATION_SEED_V13",
    "PROJECTION_INITIALIZATION_SEED_V14",
    "REMOVED_GROUND_HEAD_PARAMETER_COUNT_V14",
    "REPRESENTATION_GROUP_PARAMETER_COUNT_V13",
    "REPRESENTATION_GROUP_PARAMETER_COUNT_V14",
    "REPRESENTATION_PARAMETER_PREFIXES_V13",
    "SHARED_PARAMETER_PREFIXES_V13",
    "SHARED_ROUTE_PARAMETER_COUNT_V13",
    "SHARED_ROUTE_PARAMETER_COUNT_V14",
    "TARGET_BOTTLENECK_PARAMETER_COUNT_V13",
    "TARGET_BOTTLENECK_PARAMETER_COUNT_V14",
    "TARGET_PARAMETER_PREFIXES_V13",
    "UnifiedRaySurvivalCameraEvidenceBottleneckLiftV14",
    "UnifiedRaySurvivalCameraEvidenceHeadV14",
    "finite_ground_clear_logits_v14",
    "fractional_ray_log_survival_v14",
    "sample_hazards_at_ground_queries_v14",
]
