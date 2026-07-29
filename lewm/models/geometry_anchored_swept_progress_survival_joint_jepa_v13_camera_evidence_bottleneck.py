"""RGB Camera-evidence-bottleneck swept-progress joint JEPA V13.

V13 replaces every predecessor BEV lift with retained, pre-composition V4
Camera evidence.  Nominal geometry is a registered part of the RGB-only
model.  A separate auxiliary path may query the already-decoded dense image
features with per-frame calibration for fine Camera supervision, but it never
changes or supplies the nominal bottleneck state.
"""
from __future__ import annotations

import copy
from typing import Mapping, NamedTuple

import torch
import torch.nn as nn

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    CAMERA_HORIZONTAL_FOV_DEG,
    CAMERA_NEAR_M,
    CAMERA_VERTICAL_FOV_DEG,
    GROUND_SUPPORT_COUNT,
    OUTPUT_CELL_SIZE_M,
    OUTPUT_FORWARD_MIN_EDGE_M,
    OUTPUT_LEFT_MIN_EDGE_M,
    OUTPUT_SHAPE,
    PIXEL_RAY_SHAPE,
    SOURCE_SHAPE,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    DEFAULT_QUERY_CHUNK_SIZE,
    DEPTH_BIN_COUNT,
    DEPTH_BIN_SIZE_M,
    DEPTH_NEAR_EDGE_M,
    GroundQueryGeometryV4,
    ObservableCameraRayEvidenceV4Model,
    ObservableCameraRayEvidenceV4RawOutput,
    ordered_obstacle_first_hit_log_probabilities_v4,
)
from lewm.models.observable_camera_ray_evidence_v4_training import (
    calibrated_pixel_ray_directions_torch_v4,
)
from lewm.models.shared_observable_camera_ray_jepa_v5 import (
    ObservableCameraRayEvidenceV4Head,
)

from .geometry_anchored_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition import (
    ACTION_VOCABULARY_V1,
    FREE_CLASS_V1,
    OCCUPIED_CLASS_V1,
    SWEEP_PROGRESS_BIN_COUNT_V1,
    UNKNOWN_CLASS_V1,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredSweptProgressSurvivalJointJepaV12,
    HeightRoleNeutralDisjointTernarySemanticDecoderV12,
    SweptProgressSurvivalHeadV1,
    SweptProgressSurvivalPredictionV1,
    final_class_macro_nll_per_row,
    latent_energy_per_row,
    neutral_disjoint_ternary_log_probabilities_v12,
)


FREE_EVIDENCE_PLANE_COUNT_V13 = 40
OCCUPIED_EVIDENCE_PLANE_COUNT_V13 = 64
ROLE_PROJECTION_WIDTH_V13 = 32
CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13 = 3_392
PROJECTION_INITIALIZATION_SEED_V13 = 20_260_729
OCCUPIED_RAY_CHUNK_SIZE_V13 = 256
OCCUPIED_MASK_OFFSETS_M_V13 = (-0.05, 0.0, 0.05)
OCCUPIED_SPLAT_CANDIDATE_DELTAS_V13 = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
)

NOMINAL_CAMERA_ORIGIN_BODY_M_V13 = (0.326, 0.0, 0.043)
NOMINAL_CAMERA_BASIS_BODY_FRU_V13 = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
)
NOMINAL_GROUND_PLANE_Z_BODY_M_V13 = -0.333

SHARED_PARAMETER_PREFIXES_V13 = (
    "encoder.",
    "bev_lift.evidence_head.",
)
REPRESENTATION_PARAMETER_PREFIXES_V13 = (
    "bev_lift.free_projection.",
    "bev_lift.occupied_projection.",
    "semantic_head.",
)
PREDICTOR_PARAMETER_PREFIXES_V13 = ("predictor.",)
TARGET_PARAMETER_PREFIXES_V13 = (
    "target_encoder.",
    "target_bev_lift.evidence_head.",
    "target_bev_lift.free_projection.",
    "target_bev_lift.occupied_projection.",
)

SHARED_ROUTE_PARAMETER_COUNT_V13 = 3_105_513
REPRESENTATION_GROUP_PARAMETER_COUNT_V13 = 22_020
PREDICTOR_GROUP_PARAMETER_COUNT_V13 = 259_073
ONLINE_TRAINABLE_PARAMETER_COUNT_V13 = 3_386_606
TARGET_BOTTLENECK_PARAMETER_COUNT_V13 = 3_108_905


class CameraEvidenceBottleneckEncodingV13(NamedTuple):
    """Nominal RGB-only evidence and its sole retained BEV state."""

    latent: torch.Tensor
    nominal_evidence: ObservableCameraRayEvidenceV4RawOutput
    free_evidence_planes: torch.Tensor
    occupied_evidence_planes: torch.Tensor


class CameraEvidenceBottleneckAuxiliaryEncodingV13(NamedTuple):
    """Nominal encoding plus supervision-only calibrated evidence."""

    latent: torch.Tensor
    nominal_evidence: ObservableCameraRayEvidenceV4RawOutput
    auxiliary_evidence: ObservableCameraRayEvidenceV4RawOutput
    free_evidence_planes: torch.Tensor
    occupied_evidence_planes: torch.Tensor


class V13TrainableParameterGroups(NamedTuple):
    """The three disjoint online groups frozen by the V13 contract."""

    shared: tuple[tuple[str, nn.Parameter], ...]
    representation: tuple[tuple[str, nn.Parameter], ...]
    predictor: tuple[tuple[str, nn.Parameter], ...]


def _candidate_splat_geometry_v13(
    row_coordinate: torch.Tensor,
    column_coordinate: torch.Tensor,
) -> tuple[
    tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...],
    torch.Tensor,
]:
    """Return ordered bilinear candidates and their valid-weight normalizer."""

    if row_coordinate.shape != column_coordinate.shape or row_coordinate.ndim != 3:
        raise ValueError("V13 splat coordinates must have matching shape (B,D,R)")
    rows, columns = OUTPUT_SHAPE
    inside_extent = (
        (row_coordinate >= -0.5)
        & (row_coordinate <= rows - 0.5)
        & (column_coordinate >= -0.5)
        & (column_coordinate <= columns - 0.5)
    )
    row_low = torch.floor(row_coordinate).to(dtype=torch.long)
    column_low = torch.floor(column_coordinate).to(dtype=torch.long)
    row_fraction = row_coordinate - row_low.to(dtype=row_coordinate.dtype)
    column_fraction = column_coordinate - column_low.to(
        dtype=column_coordinate.dtype
    )

    candidates = []
    normalizer = torch.zeros_like(row_coordinate)
    for row_delta, column_delta in OCCUPIED_SPLAT_CANDIDATE_DELTAS_V13:
        row_index = row_low + row_delta
        column_index = column_low + column_delta
        row_weight = row_fraction if row_delta else 1.0 - row_fraction
        column_weight = (
            column_fraction if column_delta else 1.0 - column_fraction
        )
        raw_weight = row_weight * column_weight
        valid = (
            inside_extent
            & (row_index >= 0)
            & (row_index < rows)
            & (column_index >= 0)
            & (column_index < columns)
        )
        normalizer = normalizer + torch.where(
            valid,
            raw_weight,
            torch.zeros_like(raw_weight),
        )
        candidates.append((row_index, column_index, valid, raw_weight))
    return tuple(candidates), normalizer


def retained_occupied_evidence_planes_v13(
    hazard_logits: torch.Tensor,
    within_bin_offset_m: torch.Tensor,
    camera_origin_body_m: torch.Tensor,
    camera_basis_body_fru: torch.Tensor,
    *,
    ray_chunk_size: int = OCCUPIED_RAY_CHUNK_SIZE_V13,
) -> torch.Tensor:
    """Retain one bilinearly splatted first-hit union plane per depth bin."""

    if (
        not isinstance(hazard_logits, torch.Tensor)
        or hazard_logits.ndim != 4
        or hazard_logits.dtype != torch.float32
    ):
        raise ValueError("V13 hazards must have shape (B,64,Hray,Wray) in float32")
    if (
        not isinstance(within_bin_offset_m, torch.Tensor)
        or tuple(within_bin_offset_m.shape) != tuple(hazard_logits.shape)
        or within_bin_offset_m.dtype != torch.float32
    ):
        raise ValueError("V13 offsets must match the float32 hazards")
    if hazard_logits.device != within_bin_offset_m.device:
        raise ValueError("V13 hazards and offsets must share a device")
    batch, depth_count, ray_height, ray_width = hazard_logits.shape
    if batch < 1 or depth_count != DEPTH_BIN_COUNT:
        raise ValueError("V13 hazards require a nonempty batch and 64 depths")
    if (ray_height, ray_width) != PIXEL_RAY_SHAPE:
        raise ValueError("V13 hazards require the exact 84x112 ray lattice")
    if tuple(camera_origin_body_m.shape) != (batch, 3):
        raise ValueError("camera_origin_body_m must have shape (B,3)")
    if tuple(camera_basis_body_fru.shape) != (batch, 3, 3):
        raise ValueError("camera_basis_body_fru must have shape (B,3,3)")
    if (
        camera_origin_body_m.device != hazard_logits.device
        or camera_basis_body_fru.device != hazard_logits.device
    ):
        raise ValueError("V13 evidence and nominal geometry must share a device")
    if (
        camera_origin_body_m.dtype != torch.float32
        or camera_basis_body_fru.dtype != torch.float32
    ):
        raise ValueError("V13 nominal geometry must use exact float32")
    chunk_size = int(ray_chunk_size)
    if chunk_size != OCCUPIED_RAY_CHUNK_SIZE_V13:
        raise ValueError("V13 requires exact occupied-ray chunks of 256")

    directions = calibrated_pixel_ray_directions_torch_v4(
        camera_basis_body_fru,
        ray_shape=(ray_height, ray_width),
        dtype=torch.float32,
    ).reshape(batch, ray_height * ray_width, 3)
    ordered = ordered_obstacle_first_hit_log_probabilities_v4(hazard_logits)
    hit_probability = ordered.hit.exp().reshape(
        batch,
        depth_count,
        ray_height * ray_width,
    )
    offset = within_bin_offset_m.reshape(
        batch,
        depth_count,
        ray_height * ray_width,
    )
    bin_centres = DEPTH_NEAR_EDGE_M + (
        torch.arange(
            depth_count,
            dtype=torch.float32,
            device=hazard_logits.device,
        )
        + 0.5
    ) * DEPTH_BIN_SIZE_M
    rows, columns = OUTPUT_SHAPE
    cell_count = rows * columns
    accumulator = hazard_logits.new_zeros(batch, depth_count, cell_count)
    epsilon = torch.finfo(torch.float32).eps

    ray_count = ray_height * ray_width
    for start in range(0, ray_count, chunk_size):
        stop = min(start + chunk_size, ray_count)
        distance = bin_centres[None, :, None] + offset[:, :, start:stop]
        point_xy = camera_origin_body_m[:, None, None, :2] + (
            distance[..., None] * directions[:, None, start:stop, :2]
        )
        row_coordinate = (
            point_xy[..., 0] - OUTPUT_FORWARD_MIN_EDGE_M
        ) / OUTPUT_CELL_SIZE_M - 0.5
        column_coordinate = (
            point_xy[..., 1] - OUTPUT_LEFT_MIN_EDGE_M
        ) / OUTPUT_CELL_SIZE_M - 0.5
        candidates, normalizer = _candidate_splat_geometry_v13(
            row_coordinate,
            column_coordinate,
        )
        for row_index, column_index, valid, raw_weight in candidates:
            weight = torch.where(
                valid,
                raw_weight / normalizer.clamp_min(epsilon),
                torch.zeros_like(raw_weight),
            )
            contribution = (
                hit_probability[:, :, start:stop] * weight
            ).clamp(min=0.0, max=1.0 - epsilon)
            cell_index = row_index * columns + column_index
            log_contribution = torch.where(
                valid,
                torch.log1p(-contribution),
                torch.zeros_like(contribution),
            )
            accumulator = accumulator.scatter_add(
                2,
                cell_index.clamp(0, cell_count - 1),
                log_contribution,
            )
    result = -torch.expm1(accumulator.reshape(batch, depth_count, rows, columns))
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("V13 retained occupied evidence is nonfinite")
    return result


class CameraEvidenceBottleneckLiftV13(nn.Module):
    """V4 evidence head plus the frozen retained 40/64-plane role lift."""

    def __init__(self, n320_fit_model: ObservableCameraRayEvidenceV4Model) -> None:
        super().__init__()
        if not isinstance(n320_fit_model, ObservableCameraRayEvidenceV4Model):
            raise TypeError("n320_fit_model must be ObservableCameraRayEvidenceV4Model")
        if (
            tuple(n320_fit_model.source_shape) != SOURCE_SHAPE
            or tuple(n320_fit_model.pixel_ray_shape) != PIXEL_RAY_SHAPE
            or n320_fit_model.query_chunk_size != DEFAULT_QUERY_CHUNK_SIZE
        ):
            raise ValueError("N320 Camera evidence geometry changed")

        caller_rng = torch.random.get_rng_state().clone()
        try:
            self.evidence_head = ObservableCameraRayEvidenceV4Head(
                source_shape=SOURCE_SHAPE,
                pixel_ray_shape=PIXEL_RAY_SHAPE,
                query_chunk_size=DEFAULT_QUERY_CHUNK_SIZE,
            )
            self.migrated_evidence_state_names = (
                self.evidence_head.migrate_from_fit_model(n320_fit_model)
            )
            self.free_projection = nn.Conv2d(
                FREE_EVIDENCE_PLANE_COUNT_V13,
                ROLE_PROJECTION_WIDTH_V13,
                kernel_size=1,
                bias=True,
            )
            self.occupied_projection = nn.Conv2d(
                OCCUPIED_EVIDENCE_PLANE_COUNT_V13,
                ROLE_PROJECTION_WIDTH_V13,
                kernel_size=1,
                bias=True,
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(PROJECTION_INITIALIZATION_SEED_V13)
            nn.init.xavier_uniform_(
                self.free_projection.weight,
                gain=1.0,
                generator=generator,
            )
            nn.init.zeros_(self.free_projection.bias)
            nn.init.xavier_uniform_(
                self.occupied_projection.weight,
                gain=1.0,
                generator=generator,
            )
            nn.init.zeros_(self.occupied_projection.bias)
        finally:
            torch.random.set_rng_state(caller_rng)
        self.activation = nn.GELU(approximate="none")

        self.register_buffer(
            "nominal_camera_origin_body_m",
            torch.tensor(NOMINAL_CAMERA_ORIGIN_BODY_M_V13, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "nominal_camera_basis_body_fru",
            torch.tensor(NOMINAL_CAMERA_BASIS_BODY_FRU_V13, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "nominal_ground_plane_z_body_m",
            torch.tensor(NOMINAL_GROUND_PLANE_Z_BODY_M_V13, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "horizontal_fov_degrees",
            torch.tensor(CAMERA_HORIZONTAL_FOV_DEG, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "vertical_fov_degrees",
            torch.tensor(CAMERA_VERTICAL_FOV_DEG, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "camera_near_m",
            torch.tensor(CAMERA_NEAR_M, dtype=torch.float32),
            persistent=True,
        )

        with torch.no_grad():
            nominal_query = self._nominal_ground_query(
                batch=1,
                dtype=torch.float32,
                device=self.nominal_camera_origin_body_m.device,
            )
            free_valid = self._flatten_ground_supports(
                nominal_query.in_frustum
            ).squeeze(0).any(dim=0)
            occupied_valid = self._build_occupied_cell_valid_mask()
        self.register_buffer(
            "free_cell_valid_mask",
            free_valid.to(dtype=torch.bool).contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "occupied_cell_valid_mask",
            occupied_valid.to(dtype=torch.bool).contiguous(),
            persistent=True,
        )

        projection_count = sum(
            parameter.numel()
            for module in (self.free_projection, self.occupied_projection)
            for parameter in module.parameters()
        )
        if projection_count != CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13:
            raise RuntimeError("V13 projection parameter count changed")

    @property
    def cell_valid_mask(self) -> torch.Tensor:
        """Union of the two registered V13 role-valid masks."""

        return self.free_cell_valid_mask | self.occupied_cell_valid_mask

    def _nominal_geometry(
        self,
        *,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dtype != torch.float32:
            raise TypeError("V13 nominal evidence requires exact float32")
        if self.nominal_camera_origin_body_m.device != device:
            raise TypeError("V13 nominal geometry and evidence must share a device")
        origin = self.nominal_camera_origin_body_m[None].expand(batch, -1)
        basis = self.nominal_camera_basis_body_fru[None].expand(batch, -1, -1)
        ground = self.nominal_ground_plane_z_body_m.expand(batch)
        return origin, basis, ground

    def _nominal_ground_query(
        self,
        *,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> GroundQueryGeometryV4:
        origin, basis, ground = self._nominal_geometry(
            batch=batch,
            dtype=dtype,
            device=device,
        )
        return self.evidence_head.ground_query_geometry(origin, basis, ground)

    @staticmethod
    def _flatten_ground_supports(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or tuple(value.shape[1:]) != (
            SOURCE_SHAPE[0],
            SOURCE_SHAPE[1],
            GROUND_SUPPORT_COUNT,
        ):
            raise ValueError("ground evidence must have shape (B,128,128,5)")
        batch = value.shape[0]
        return (
            value.reshape(batch, OUTPUT_SHAPE[0], 2, OUTPUT_SHAPE[1], 2, 5)
            .permute(0, 2, 4, 5, 1, 3)
            .reshape(batch, 20, *OUTPUT_SHAPE)
            .contiguous()
        )

    def free_evidence_planes(
        self,
        nominal_evidence: ObservableCameraRayEvidenceV4RawOutput,
    ) -> torch.Tensor:
        """Return ordered raw ground logits followed by fixed validity bits."""

        if not isinstance(nominal_evidence, ObservableCameraRayEvidenceV4RawOutput):
            raise TypeError("nominal_evidence must be a V4 raw output")
        logits = self._flatten_ground_supports(
            nominal_evidence.ground_clear_to_target_logits
        )
        valid = self._flatten_ground_supports(
            nominal_evidence.ground_query_in_frustum
        )
        logits = torch.where(valid, logits, torch.zeros_like(logits))
        result = torch.cat((logits, valid.to(dtype=logits.dtype)), dim=1)
        if tuple(result.shape[1:]) != (
            FREE_EVIDENCE_PLANE_COUNT_V13,
            *OUTPUT_SHAPE,
        ):
            raise RuntimeError("V13 FREE evidence-plane shape changed")
        return result

    def occupied_evidence_planes(
        self,
        nominal_evidence: ObservableCameraRayEvidenceV4RawOutput,
    ) -> torch.Tensor:
        """Return the 64 retained nominal first-hit depth planes."""

        if not isinstance(nominal_evidence, ObservableCameraRayEvidenceV4RawOutput):
            raise TypeError("nominal_evidence must be a V4 raw output")
        batch = nominal_evidence.pixel_first_hit_hazard_logits.shape[0]
        origin, basis, _ = self._nominal_geometry(
            batch=batch,
            dtype=nominal_evidence.pixel_first_hit_hazard_logits.dtype,
            device=nominal_evidence.pixel_first_hit_hazard_logits.device,
        )
        result = retained_occupied_evidence_planes_v13(
            nominal_evidence.pixel_first_hit_hazard_logits,
            nominal_evidence.pixel_within_bin_offset_m,
            origin,
            basis,
            ray_chunk_size=OCCUPIED_RAY_CHUNK_SIZE_V13,
        )
        if tuple(result.shape[1:]) != (
            OCCUPIED_EVIDENCE_PLANE_COUNT_V13,
            *OUTPUT_SHAPE,
        ):
            raise RuntimeError("V13 OCCUPIED evidence-plane shape changed")
        return result

    @staticmethod
    def _raw_evidence(
        hazard: torch.Tensor,
        offset: torch.Tensor,
        ground_logits: torch.Tensor,
        query: GroundQueryGeometryV4,
    ) -> ObservableCameraRayEvidenceV4RawOutput:
        return ObservableCameraRayEvidenceV4RawOutput(
            pixel_first_hit_hazard_logits=hazard,
            pixel_within_bin_offset_m=offset,
            ground_clear_to_target_logits=ground_logits,
            ground_query_in_frustum=query.in_frustum,
            ground_query_uv_px=query.uv_px,
            ground_target_distance_m=query.target_distance_m,
        )

    def _latent_from_nominal_evidence(
        self,
        nominal_evidence: ObservableCameraRayEvidenceV4RawOutput,
    ) -> CameraEvidenceBottleneckEncodingV13:
        free_planes = self.free_evidence_planes(nominal_evidence)
        occupied_planes = self.occupied_evidence_planes(nominal_evidence)
        latent = torch.cat(
            (
                self.activation(self.free_projection(free_planes)),
                self.activation(self.occupied_projection(occupied_planes)),
            ),
            dim=1,
        )
        if tuple(latent.shape[1:]) != (64, *OUTPUT_SHAPE):
            raise RuntimeError("V13 Camera-evidence bottleneck shape changed")
        if not bool(torch.isfinite(latent).all()):
            raise FloatingPointError("V13 Camera-evidence bottleneck is nonfinite")
        return CameraEvidenceBottleneckEncodingV13(
            latent=latent,
            nominal_evidence=nominal_evidence,
            free_evidence_planes=free_planes,
            occupied_evidence_planes=occupied_planes,
        )

    def forward_with_evidence(
        self,
        patch_tokens: torch.Tensor,
    ) -> CameraEvidenceBottleneckEncodingV13:
        """Decode nominal evidence once and construct the sole V13 state."""

        dense = self.evidence_head.decode_dense_features(patch_tokens)
        nominal_query = self._nominal_ground_query(
            batch=patch_tokens.shape[0],
            dtype=patch_tokens.dtype,
            device=patch_tokens.device,
        )
        hazard, offset = self.evidence_head.pixel_branch(dense)
        ground = self.evidence_head.ground_branch(
            dense,
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
        """Decode once; keep calibration confined to the auxiliary query."""

        dense = self.evidence_head.decode_dense_features(patch_tokens)
        nominal_query = self._nominal_ground_query(
            batch=patch_tokens.shape[0],
            dtype=patch_tokens.dtype,
            device=patch_tokens.device,
        )
        hazard, offset = self.evidence_head.pixel_branch(dense)
        nominal_ground = self.evidence_head.ground_branch(
            dense,
            nominal_query,
            query_chunk_size=self.evidence_head.query_chunk_size,
        )
        nominal = self._raw_evidence(
            hazard,
            offset,
            nominal_ground,
            nominal_query,
        )
        encoded = self._latent_from_nominal_evidence(nominal)

        auxiliary_query = self.evidence_head.ground_query_geometry(
            camera_origin_body_m,
            camera_basis_body_fru,
            ground_plane_z_body_m,
        )
        if auxiliary_query.in_frustum.shape[0] != patch_tokens.shape[0]:
            raise ValueError("patch-token and auxiliary calibration batches differ")
        auxiliary_ground = self.evidence_head.ground_branch(
            dense,
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

    def _build_occupied_cell_valid_mask(self) -> torch.Tensor:
        origin, basis, _ = self._nominal_geometry(
            batch=1,
            dtype=torch.float32,
            device=self.nominal_camera_origin_body_m.device,
        )
        ray_height, ray_width = PIXEL_RAY_SHAPE
        directions = calibrated_pixel_ray_directions_torch_v4(
            basis,
            ray_shape=PIXEL_RAY_SHAPE,
            dtype=torch.float32,
        ).reshape(1, ray_height * ray_width, 3)
        bin_centres = DEPTH_NEAR_EDGE_M + (
            torch.arange(DEPTH_BIN_COUNT, dtype=torch.float32) + 0.5
        ) * DEPTH_BIN_SIZE_M
        rows, columns = OUTPUT_SHAPE
        result = torch.zeros(rows * columns, dtype=torch.bool)
        ray_count = ray_height * ray_width
        for offset_m in OCCUPIED_MASK_OFFSETS_M_V13:
            for start in range(0, ray_count, OCCUPIED_RAY_CHUNK_SIZE_V13):
                stop = min(start + OCCUPIED_RAY_CHUNK_SIZE_V13, ray_count)
                distance = bin_centres[None, :, None] + float(offset_m)
                point_xy = origin[:, None, None, :2] + (
                    distance[..., None] * directions[:, None, start:stop, :2]
                )
                row_coordinate = (
                    point_xy[..., 0] - OUTPUT_FORWARD_MIN_EDGE_M
                ) / OUTPUT_CELL_SIZE_M - 0.5
                column_coordinate = (
                    point_xy[..., 1] - OUTPUT_LEFT_MIN_EDGE_M
                ) / OUTPUT_CELL_SIZE_M - 0.5
                candidates, _ = _candidate_splat_geometry_v13(
                    row_coordinate,
                    column_coordinate,
                )
                for row_index, column_index, valid, _ in candidates:
                    flat_index = row_index * columns + column_index
                    result[flat_index[valid]] = True
        return result.reshape(rows, columns)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_with_evidence(patch_tokens).latent


class GeometryAnchoredSweptProgressSurvivalJointJepaV13(
    GeometryAnchoredSweptProgressSurvivalJointJepaV12
):
    """V12 semantics/predictor over the sole retained Camera-evidence state."""

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
        super().__init__(n320_encoder_state_dict, sweep_masks, config)
        if int(self.target_hard_sync_count.item()) != 1:
            raise RuntimeError("predecessor construction hard-sync count changed")
        if int(self.ema_update_count.item()) != 0:
            raise RuntimeError("predecessor construction EMA count changed")

        self.bev_lift = CameraEvidenceBottleneckLiftV13(n320_fit_model)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_bev_lift = copy.deepcopy(self.bev_lift)
        self.target_hard_sync_count.fill_(1)
        self.ema_update_count.zero_()
        self._freeze_target()
        self._assert_final_target_identity()
        self._assert_parameter_accounting()

    def _assert_final_target_identity(self) -> None:
        for online, target in zip(
            self.online_target_modules(),
            self.target_modules(),
            strict=True,
        ):
            online_state = online.state_dict()
            target_state = target.state_dict()
            if online_state.keys() != target_state.keys() or any(
                not torch.equal(value, target_state[name])
                for name, value in online_state.items()
            ):
                raise RuntimeError("V13 final target hard-sync identity failed")
        if int(self.target_hard_sync_count.item()) != 1:
            raise RuntimeError("V13 must account exactly one final hard sync")
        if int(self.ema_update_count.item()) != 0:
            raise RuntimeError("V13 must begin before any EMA update")

    def trainable_parameter_groups_v13(self) -> V13TrainableParameterGroups:
        """Return and validate the three exact disjoint online groups."""

        named = tuple(
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )

        def select(prefixes: tuple[str, ...]) -> tuple[tuple[str, nn.Parameter], ...]:
            return tuple(
                (name, parameter)
                for name, parameter in named
                if name.startswith(prefixes)
            )

        groups = V13TrainableParameterGroups(
            shared=select(SHARED_PARAMETER_PREFIXES_V13),
            representation=select(REPRESENTATION_PARAMETER_PREFIXES_V13),
            predictor=select(PREDICTOR_PARAMETER_PREFIXES_V13),
        )
        selected_names = [name for group in groups for name, _ in group]
        if len(selected_names) != len(set(selected_names)):
            raise RuntimeError("V13 trainable parameter groups overlap")
        if set(selected_names) != {name for name, _ in named}:
            raise RuntimeError("V13 trainable parameter groups do not cover the model")
        return groups

    def _assert_parameter_accounting(self) -> None:
        groups = self.trainable_parameter_groups_v13()
        counts = tuple(
            sum(parameter.numel() for _, parameter in group) for group in groups
        )
        if counts != (
            SHARED_ROUTE_PARAMETER_COUNT_V13,
            REPRESENTATION_GROUP_PARAMETER_COUNT_V13,
            PREDICTOR_GROUP_PARAMETER_COUNT_V13,
        ):
            raise RuntimeError(f"V13 online parameter-group counts changed: {counts}")
        if sum(counts) != ONLINE_TRAINABLE_PARAMETER_COUNT_V13:
            raise RuntimeError("V13 total online trainable parameter count changed")
        target_count = sum(
            parameter.numel()
            for module in self.target_modules()
            for parameter in module.parameters()
        )
        if target_count != TARGET_BOTTLENECK_PARAMETER_COUNT_V13:
            raise RuntimeError("V13 target bottleneck parameter count changed")

    def encode_online_with_evidence(
        self,
        rgb: torch.Tensor,
    ) -> CameraEvidenceBottleneckEncodingV13:
        self._validate_rgb(rgb, name="online_rgb")
        patch_tokens = self.encoder.forward_tokens(rgb)[:, 1:]
        return self.bev_lift.forward_with_evidence(patch_tokens)

    def encode_online_with_sampling(self, rgb: torch.Tensor) -> None:
        """Reject the inherited V10/V11 sampling schema removed by V13."""

        del rgb
        raise RuntimeError(
            "V13 removed encode_online_with_sampling; "
            "use encode_online_with_evidence"
        )

    @torch.no_grad()
    def encode_target_with_sampling(self, rgb: torch.Tensor) -> None:
        """Reject the inherited V10/V11 target-sampling schema removed by V13."""

        del rgb
        raise RuntimeError(
            "V13 removed encode_target_with_sampling; use encode_target"
        )

    def encode_online_with_auxiliary_evidence(
        self,
        rgb: torch.Tensor,
        *,
        camera_origin_body_m: torch.Tensor,
        camera_basis_body_fru: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
    ) -> CameraEvidenceBottleneckAuxiliaryEncodingV13:
        self._validate_rgb(rgb, name="online_rgb")
        patch_tokens = self.encoder.forward_tokens(rgb)[:, 1:]
        return self.bev_lift.forward_with_auxiliary_evidence(
            patch_tokens,
            camera_origin_body_m=camera_origin_body_m,
            camera_basis_body_fru=camera_basis_body_fru,
            ground_plane_z_body_m=ground_plane_z_body_m,
        )

    def encode_online_training(
        self,
        rgb: torch.Tensor,
        *,
        camera_origin_body_m: torch.Tensor,
        camera_basis_body_fru: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
    ) -> CameraEvidenceBottleneckAuxiliaryEncodingV13:
        """Alias spelling the only calibration-bearing training path."""

        return self.encode_online_with_auxiliary_evidence(
            rgb,
            camera_origin_body_m=camera_origin_body_m,
            camera_basis_body_fru=camera_basis_body_fru,
            ground_plane_z_body_m=ground_plane_z_body_m,
        )

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        expected = (self.config.bev_dim, *self.config.bev_size)
        if latent.ndim != 4 or tuple(latent.shape[1:]) != expected:
            raise ValueError(f"latent must have shape (B,{expected})")
        free, occupied = self.semantic_head.evidence_logits(latent)
        free_valid = self.bev_lift.free_cell_valid_mask[None].expand(
            latent.shape[0], -1, -1
        )
        occupied_valid = self.bev_lift.occupied_cell_valid_mask[None].expand(
            latent.shape[0], -1, -1
        )
        free = torch.where(free_valid, free, torch.full_like(free, -20.0))
        occupied = torch.where(
            occupied_valid,
            occupied,
            torch.full_like(occupied, -20.0),
        )
        logits = neutral_disjoint_ternary_log_probabilities_v12(free, occupied)
        valid = (free_valid | occupied_valid)[:, None]
        invalid_logits = logits.new_tensor((0.0, -20.0, -20.0))[
            None, :, None, None
        ]
        return torch.where(valid, logits, invalid_logits)


CameraEvidenceBottleneckJointJepaV13 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV13
)
GeometryAnchoredSweptProgressSurvivalJointJepaV13Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV13
)
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_VOCABULARY_V1",
    "CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13",
    "CameraEvidenceBottleneckAuxiliaryEncodingV13",
    "CameraEvidenceBottleneckEncodingV13",
    "CameraEvidenceBottleneckJointJepaV13",
    "CameraEvidenceBottleneckLiftV13",
    "FREE_CLASS_V1",
    "FREE_EVIDENCE_PLANE_COUNT_V13",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1Config",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV13",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV13Config",
    "HeightRoleNeutralDisjointTernarySemanticDecoderV12",
    "NOMINAL_CAMERA_BASIS_BODY_FRU_V13",
    "NOMINAL_CAMERA_ORIGIN_BODY_M_V13",
    "NOMINAL_GROUND_PLANE_Z_BODY_M_V13",
    "OCCUPIED_CLASS_V1",
    "OCCUPIED_EVIDENCE_PLANE_COUNT_V13",
    "OCCUPIED_MASK_OFFSETS_M_V13",
    "OCCUPIED_RAY_CHUNK_SIZE_V13",
    "OCCUPIED_SPLAT_CANDIDATE_DELTAS_V13",
    "ONLINE_TRAINABLE_PARAMETER_COUNT_V13",
    "PREDICTOR_GROUP_PARAMETER_COUNT_V13",
    "PREDICTOR_PARAMETER_PREFIXES_V13",
    "PROJECTION_INITIALIZATION_SEED_V13",
    "REPRESENTATION_GROUP_PARAMETER_COUNT_V13",
    "REPRESENTATION_PARAMETER_PREFIXES_V13",
    "ROLE_PROJECTION_WIDTH_V13",
    "SHARED_PARAMETER_PREFIXES_V13",
    "SHARED_ROUTE_PARAMETER_COUNT_V13",
    "SWEEP_PROGRESS_BIN_COUNT_V1",
    "SweptProgressSurvivalHeadV1",
    "SweptProgressSurvivalPredictionV1",
    "TARGET_BOTTLENECK_PARAMETER_COUNT_V13",
    "TARGET_PARAMETER_PREFIXES_V13",
    "UNKNOWN_CLASS_V1",
    "V13TrainableParameterGroups",
    "final_class_macro_nll_per_row",
    "final_class_macro_nll_per_row_v1",
    "latent_energy_per_row",
    "latent_energy_per_row_v1",
    "neutral_disjoint_ternary_log_probabilities_v12",
    "retained_occupied_evidence_planes_v13",
]
