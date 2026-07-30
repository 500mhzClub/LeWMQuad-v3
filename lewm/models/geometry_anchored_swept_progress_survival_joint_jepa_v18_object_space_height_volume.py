"""Object-space height-volume RGB joint JEPA V18.

V18 retains V14's jointly learned ordered Camera ray field, but replaces the
2-D FREE/OCCUPIED evidence projections with one registered 3-D body-frame
volume.  The eight height slices remain distinct in the sole latent presented
to the EMA target and action-conditioned predictor.
"""
from __future__ import annotations

import copy
import math
from typing import Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    CAMERA_HORIZONTAL_FOV_DEG,
    CAMERA_VERTICAL_FOV_DEG,
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

from .geometry_anchored_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition import (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredSweptProgressSurvivalJointJepaV12,
    neutral_disjoint_ternary_log_probabilities_v12,
)
from .geometry_anchored_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck import (
    ACTION_VOCABULARY_V1,
    FREE_CLASS_V1,
    OCCUPIED_CLASS_V1,
    SWEEP_PROGRESS_BIN_COUNT_V1,
    UNKNOWN_CLASS_V1,
    NOMINAL_CAMERA_BASIS_BODY_FRU_V13,
    NOMINAL_CAMERA_ORIGIN_BODY_M_V13,
    NOMINAL_GROUND_PLANE_Z_BODY_M_V13,
    PREDICTOR_GROUP_PARAMETER_COUNT_V13 as _PREDICTOR_PARAMETER_COUNT,
    V13TrainableParameterGroups,
    final_class_macro_nll_per_row,
    latent_energy_per_row,
)
from .geometry_anchored_swept_progress_survival_joint_jepa_v14_unified_ray_survival import (
    MIGRATED_EVIDENCE_STATE_COUNT_V14,
    UnifiedRaySurvivalCameraEvidenceHeadV14,
)


HEIGHT_CENTRES_M_V18 = (
    -0.333,
    -0.183,
    -0.033,
    0.117,
    0.267,
    0.417,
    0.567,
    0.717,
)
HEIGHT_COUNT_V18 = 8
VOXEL_INPUT_CHANNEL_COUNT_V18 = 5
VOLUME_CHANNEL_COUNT_V18 = 8
FLATTENED_VOLUME_CHANNEL_COUNT_V18 = 64
VOLUME_INITIALIZATION_SEED_V18 = 20_260_729
PROJECTION_INITIALIZATION_SEED_V18 = VOLUME_INITIALIZATION_SEED_V18

VOXEL_MIN_FORWARD_M_V18 = -0.95
VOXEL_MAX_FORWARD_M_V18 = 5.35
VOXEL_MIN_LEFT_M_V18 = -3.15
VOXEL_MAX_LEFT_M_V18 = 3.15
VOXEL_MIN_CAMERA_FORWARD_M_V18 = 0.05
VOXEL_MIN_RANGE_M_V18 = 0.05
VOXEL_MAX_RANGE_M_V18 = 6.45
HEIGHT_NORMALIZATION_CENTRE_M_V18 = 0.192
HEIGHT_NORMALIZATION_HALF_RANGE_M_V18 = 0.525
OFFSET_NORMALIZATION_M_V18 = 0.05

OBJECT_SPACE_HEIGHT_VOLUME_PARAMETER_COUNT_V18 = 3_520
OBJECT_SPACE_HEIGHT_VOLUME_SEMANTIC_PARAMETER_COUNT_V18 = 73_986
SHARED_ROUTE_PARAMETER_COUNT_V18 = 3_102_824
REPRESENTATION_GROUP_PARAMETER_COUNT_V18 = 77_506
PREDICTOR_GROUP_PARAMETER_COUNT_V18 = _PREDICTOR_PARAMETER_COUNT
ONLINE_TRAINABLE_PARAMETER_COUNT_V18 = 3_439_403
TARGET_BOTTLENECK_PARAMETER_COUNT_V18 = 3_106_344

SHARED_PARAMETER_PREFIXES_V18 = (
    "encoder.",
    "bev_lift.evidence_head.",
)
REPRESENTATION_PARAMETER_PREFIXES_V18 = (
    "bev_lift.point_projection.",
    "bev_lift.volume_block.",
    "semantic_head.",
)
PREDICTOR_PARAMETER_PREFIXES_V18 = ("predictor.",)
TARGET_PARAMETER_PREFIXES_V18 = (
    "target_encoder.",
    "target_bev_lift.evidence_head.",
    "target_bev_lift.point_projection.",
    "target_bev_lift.volume_block.",
)

# The inherited V13 executor indexes these spellings.  They intentionally bind
# the V18 inventory rather than preserving the retired V13 projection counts.
CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13 = (
    OBJECT_SPACE_HEIGHT_VOLUME_PARAMETER_COUNT_V18
)
SHARED_ROUTE_PARAMETER_COUNT_V13 = SHARED_ROUTE_PARAMETER_COUNT_V18
REPRESENTATION_GROUP_PARAMETER_COUNT_V13 = REPRESENTATION_GROUP_PARAMETER_COUNT_V18
PREDICTOR_GROUP_PARAMETER_COUNT_V13 = PREDICTOR_GROUP_PARAMETER_COUNT_V18
ONLINE_TRAINABLE_PARAMETER_COUNT_V13 = ONLINE_TRAINABLE_PARAMETER_COUNT_V18
TARGET_BOTTLENECK_PARAMETER_COUNT_V13 = TARGET_BOTTLENECK_PARAMETER_COUNT_V18
PROJECTION_INITIALIZATION_SEED_V13 = PROJECTION_INITIALIZATION_SEED_V18
SHARED_PARAMETER_PREFIXES_V13 = SHARED_PARAMETER_PREFIXES_V18
REPRESENTATION_PARAMETER_PREFIXES_V13 = REPRESENTATION_PARAMETER_PREFIXES_V18
PREDICTOR_PARAMETER_PREFIXES_V13 = PREDICTOR_PARAMETER_PREFIXES_V18
TARGET_PARAMETER_PREFIXES_V13 = TARGET_PARAMETER_PREFIXES_V18


class ObjectSpaceHeightVolumeGeometryV18(NamedTuple):
    """Fixed nominal body-frame voxel geometry in registered Z/Y/X order."""

    voxel_xyz_body_m: torch.Tensor
    sample_grid_xyz: torch.Tensor
    voxel_visible: torch.Tensor
    normalized_registered_height: torch.Tensor


class ObjectSpaceHeightVolumeEncodingV18(NamedTuple):
    """Nominal Camera evidence and the sole retained V18 JEPA state."""

    latent: torch.Tensor
    nominal_evidence: ObservableCameraRayEvidenceV4RawOutput
    voxel_inputs: torch.Tensor
    voxel_visible: torch.Tensor
    height_volume: torch.Tensor


class ObjectSpaceHeightVolumeAuxiliaryEncodingV18(NamedTuple):
    """Nominal V18 state plus supervision-only calibrated Camera evidence."""

    latent: torch.Tensor
    nominal_evidence: ObservableCameraRayEvidenceV4RawOutput
    auxiliary_evidence: ObservableCameraRayEvidenceV4RawOutput
    voxel_inputs: torch.Tensor
    voxel_visible: torch.Tensor
    height_volume: torch.Tensor


def object_space_voxel_geometry_v18() -> ObjectSpaceHeightVolumeGeometryV18:
    """Construct the exact fixed nominal V18 voxel projection geometry."""

    rows, columns = OUTPUT_SHAPE
    forward = OUTPUT_FORWARD_MIN_EDGE_M + (
        torch.arange(rows, dtype=torch.float32) + 0.5
    ) * OUTPUT_CELL_SIZE_M
    left = OUTPUT_LEFT_MIN_EDGE_M + (
        torch.arange(columns, dtype=torch.float32) + 0.5
    ) * OUTPUT_CELL_SIZE_M
    height = torch.tensor(HEIGHT_CENTRES_M_V18, dtype=torch.float32)
    if (
        rows != 64
        or columns != 64
        or not torch.allclose(
            forward[[0, -1]],
            torch.tensor(
                (VOXEL_MIN_FORWARD_M_V18, VOXEL_MAX_FORWARD_M_V18),
                dtype=torch.float32,
            ),
            rtol=0.0,
            atol=1.0e-6,
        )
        or not torch.allclose(
            left[[0, -1]],
            torch.tensor(
                (VOXEL_MIN_LEFT_M_V18, VOXEL_MAX_LEFT_M_V18),
                dtype=torch.float32,
            ),
            rtol=0.0,
            atol=1.0e-6,
        )
    ):
        raise RuntimeError("V18 registered XY grid changed")

    z, x, y = torch.meshgrid(height, forward, left, indexing="ij")
    xyz = torch.stack((x, y, z), dim=-1).contiguous()
    origin = torch.tensor(NOMINAL_CAMERA_ORIGIN_BODY_M_V13, dtype=torch.float32)
    basis = torch.tensor(NOMINAL_CAMERA_BASIS_BODY_FRU_V13, dtype=torch.float32)
    delta = xyz - origin
    camera_forward = torch.sum(delta * basis[0], dim=-1)
    camera_right = torch.sum(delta * basis[1], dim=-1)
    camera_up = torch.sum(delta * basis[2], dim=-1)
    range_m = torch.linalg.vector_norm(delta, dim=-1)

    horizontal_tangent = math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEG) * 0.5)
    vertical_tangent = math.tan(math.radians(CAMERA_VERTICAL_FOV_DEG) * 0.5)
    grid_x = camera_right / (camera_forward * horizontal_tangent)
    grid_y = -camera_up / (camera_forward * vertical_tangent)
    grid_z = 2.0 * (
        range_m - DEPTH_NEAR_EDGE_M
    ) / (DEPTH_BIN_COUNT * DEPTH_BIN_SIZE_M) - 1.0
    visible = (
        (camera_forward >= VOXEL_MIN_CAMERA_FORWARD_M_V18)
        & (grid_x.abs() <= 1.0)
        & (grid_y.abs() <= 1.0)
        & (range_m >= VOXEL_MIN_RANGE_M_V18)
        & (range_m <= VOXEL_MAX_RANGE_M_V18)
    )
    normalized_height = (
        z - HEIGHT_NORMALIZATION_CENTRE_M_V18
    ) / HEIGHT_NORMALIZATION_HALF_RANGE_M_V18
    sample_grid = torch.stack((grid_x, grid_y, grid_z), dim=-1).contiguous()
    if not bool(torch.isfinite(sample_grid).all()):
        raise FloatingPointError("V18 voxel sample geometry is nonfinite")
    if tuple(xyz.shape) != (HEIGHT_COUNT_V18, rows, columns, 3):
        raise RuntimeError("V18 voxel geometry shape changed")
    return ObjectSpaceHeightVolumeGeometryV18(
        voxel_xyz_body_m=xyz,
        sample_grid_xyz=sample_grid,
        voxel_visible=visible.contiguous(),
        normalized_registered_height=normalized_height.contiguous(),
    )


def ordered_ray_volume_source_v18(
    hazard_logits: torch.Tensor,
    within_bin_offset_m: torch.Tensor,
) -> torch.Tensor:
    """Return ordered hit, bin-centre clear, and normalized-offset channels."""

    if (
        not isinstance(hazard_logits, torch.Tensor)
        or hazard_logits.ndim != 4
        or tuple(hazard_logits.shape[1:])
        != (DEPTH_BIN_COUNT, *PIXEL_RAY_SHAPE)
        or hazard_logits.dtype != torch.float32
    ):
        raise ValueError("V18 hazards must have float32 shape (B,64,84,112)")
    if (
        not isinstance(within_bin_offset_m, torch.Tensor)
        or tuple(within_bin_offset_m.shape) != tuple(hazard_logits.shape)
        or within_bin_offset_m.dtype != torch.float32
        or within_bin_offset_m.device != hazard_logits.device
    ):
        raise ValueError("V18 offsets must match the float32 hazards")
    if hazard_logits.shape[0] < 1:
        raise ValueError("V18 ray source requires a nonempty batch")

    ordered = ordered_obstacle_first_hit_log_probabilities_v4(hazard_logits)
    log_survive = F.logsigmoid(-hazard_logits)
    exclusive_survival = torch.cat(
        (
            torch.zeros_like(log_survive[:, :1]),
            torch.cumsum(log_survive, dim=1)[:, :-1],
        ),
        dim=1,
    )
    first_hit_probability = ordered.hit.exp()
    clear_to_bin_centre_probability = torch.exp(
        exclusive_survival + 0.5 * log_survive
    )
    normalized_offset = within_bin_offset_m / OFFSET_NORMALIZATION_M_V18
    source = torch.stack(
        (
            first_hit_probability,
            clear_to_bin_centre_probability,
            normalized_offset,
        ),
        dim=1,
    )
    if tuple(source.shape[1:]) != (3, DEPTH_BIN_COUNT, *PIXEL_RAY_SHAPE):
        raise RuntimeError("V18 ordered ray-volume source shape changed")
    if not bool(torch.isfinite(source).all()):
        raise FloatingPointError("V18 ordered ray-volume source is nonfinite")
    return source


def flatten_height_major_volume_v18(volume: torch.Tensor) -> torch.Tensor:
    """Flatten (channel,height) with height-major/channel-minor ordering."""

    if (
        not isinstance(volume, torch.Tensor)
        or volume.ndim != 5
        or tuple(volume.shape[1:])
        != (VOLUME_CHANNEL_COUNT_V18, HEIGHT_COUNT_V18, *OUTPUT_SHAPE)
    ):
        raise ValueError("volume must have shape (B,8,8,64,64)")
    return (
        volume.permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(volume.shape[0], FLATTENED_VOLUME_CHANNEL_COUNT_V18, *OUTPUT_SHAPE)
    )


def unflatten_height_major_volume_v18(latent: torch.Tensor) -> torch.Tensor:
    """Invert :func:`flatten_height_major_volume_v18` exactly."""

    if (
        not isinstance(latent, torch.Tensor)
        or latent.ndim != 4
        or tuple(latent.shape[1:])
        != (FLATTENED_VOLUME_CHANNEL_COUNT_V18, *OUTPUT_SHAPE)
    ):
        raise ValueError("latent must have shape (B,64,64,64)")
    return (
        latent.reshape(
            latent.shape[0],
            HEIGHT_COUNT_V18,
            VOLUME_CHANNEL_COUNT_V18,
            *OUTPUT_SHAPE,
        )
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


class ObjectSpaceHeightVolumeResidualBlockV18(nn.Module):
    """The sole two-convolution local 3-D residual block."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            VOLUME_CHANNEL_COUNT_V18,
            VOLUME_CHANNEL_COUNT_V18,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv3d(
            VOLUME_CHANNEL_COUNT_V18,
            VOLUME_CHANNEL_COUNT_V18,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.activation = nn.GELU(approximate="none")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.conv2(self.activation(self.conv1(value)))


class ObjectSpaceHeightVolumeLiftV18(nn.Module):
    """Project V14's shared ordered Camera field into one 3-D JEPA state."""

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
            self.evidence_head = UnifiedRaySurvivalCameraEvidenceHeadV14(
                source_shape=SOURCE_SHAPE,
                pixel_ray_shape=PIXEL_RAY_SHAPE,
                query_chunk_size=DEFAULT_QUERY_CHUNK_SIZE,
            )
            self.migrated_evidence_state_names = (
                self.evidence_head.migrate_from_fit_model(n320_fit_model)
            )
            self.point_projection = nn.Conv3d(
                VOXEL_INPUT_CHANNEL_COUNT_V18,
                VOLUME_CHANNEL_COUNT_V18,
                kernel_size=1,
                bias=True,
            )
            self.volume_block = ObjectSpaceHeightVolumeResidualBlockV18()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(VOLUME_INITIALIZATION_SEED_V18)
            for layer in (
                self.point_projection,
                self.volume_block.conv1,
                self.volume_block.conv2,
            ):
                nn.init.xavier_uniform_(
                    layer.weight,
                    gain=1.0,
                    generator=generator,
                )
                nn.init.zeros_(layer.bias)
        finally:
            torch.random.set_rng_state(caller_rng)

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
        geometry = object_space_voxel_geometry_v18()
        self.register_buffer(
            "voxel_xyz_body_m",
            geometry.voxel_xyz_body_m,
            persistent=True,
        )
        self.register_buffer(
            "voxel_sample_grid_xyz",
            geometry.sample_grid_xyz,
            persistent=True,
        )
        self.register_buffer(
            "voxel_visible_mask",
            geometry.voxel_visible,
            persistent=True,
        )
        self.register_buffer(
            "normalized_registered_height",
            geometry.normalized_registered_height,
            persistent=True,
        )
        self.register_buffer(
            "cell_valid_mask",
            geometry.voxel_visible.any(dim=0).contiguous(),
            persistent=True,
        )

        learned_count = sum(
            parameter.numel()
            for module in (self.point_projection, self.volume_block)
            for parameter in module.parameters()
        )
        if learned_count != OBJECT_SPACE_HEIGHT_VOLUME_PARAMETER_COUNT_V18:
            raise RuntimeError("V18 height-volume parameter count changed")

    @property
    def free_cell_valid_mask(self) -> torch.Tensor:
        """Compatibility view: both evidence axes use the same V18 mask."""

        return self.cell_valid_mask

    @property
    def occupied_cell_valid_mask(self) -> torch.Tensor:
        """Compatibility view: both evidence axes use the same V18 mask."""

        return self.cell_valid_mask

    def _nominal_geometry(
        self,
        *,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dtype != torch.float32:
            raise TypeError("V18 nominal evidence requires exact float32")
        if self.nominal_camera_origin_body_m.device != device:
            raise TypeError("V18 nominal geometry and evidence must share a device")
        return (
            self.nominal_camera_origin_body_m[None].expand(batch, -1),
            self.nominal_camera_basis_body_fru[None].expand(batch, -1, -1),
            self.nominal_ground_plane_z_body_m.expand(batch),
        )

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

    def voxel_inputs_from_ray_field(
        self,
        hazard_logits: torch.Tensor,
        within_bin_offset_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample exactly once, then append fixed visibility and height."""

        source = ordered_ray_volume_source_v18(
            hazard_logits,
            within_bin_offset_m,
        )
        batch = source.shape[0]
        grid = self.voxel_sample_grid_xyz[None].expand(batch, -1, -1, -1, -1)
        # ``bilinear`` is PyTorch's trilinear sampler for five-dimensional input.
        sampled = F.grid_sample(
            source,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        visible = self.voxel_visible_mask[None].expand(batch, -1, -1, -1)
        visible_channel = visible[:, None].to(dtype=sampled.dtype)
        sampled = sampled * visible_channel
        height = self.normalized_registered_height[None, None].expand(
            batch, 1, -1, -1, -1
        )
        height = height * visible_channel
        voxel_inputs = torch.cat((sampled, visible_channel, height), dim=1)
        if tuple(voxel_inputs.shape[1:]) != (
            VOXEL_INPUT_CHANNEL_COUNT_V18,
            HEIGHT_COUNT_V18,
            *OUTPUT_SHAPE,
        ):
            raise RuntimeError("V18 voxel-input shape changed")
        if not bool(torch.isfinite(voxel_inputs).all()):
            raise FloatingPointError("V18 voxel inputs are nonfinite")
        return voxel_inputs, visible

    def _latent_from_nominal_evidence(
        self,
        nominal_evidence: ObservableCameraRayEvidenceV4RawOutput,
    ) -> ObjectSpaceHeightVolumeEncodingV18:
        voxel_inputs, visible = self.voxel_inputs_from_ray_field(
            nominal_evidence.pixel_first_hit_hazard_logits,
            nominal_evidence.pixel_within_bin_offset_m,
        )
        mask = visible[:, None].to(dtype=voxel_inputs.dtype)
        projected = self.point_projection(voxel_inputs) * mask
        volume = self.volume_block(projected) * mask
        latent = flatten_height_major_volume_v18(volume)
        if not bool(torch.isfinite(volume).all()) or not bool(
            torch.isfinite(latent).all()
        ):
            raise FloatingPointError("V18 object-space height volume is nonfinite")
        return ObjectSpaceHeightVolumeEncodingV18(
            latent=latent,
            nominal_evidence=nominal_evidence,
            voxel_inputs=voxel_inputs,
            voxel_visible=visible,
            height_volume=volume,
        )

    def forward_with_evidence(
        self,
        patch_tokens: torch.Tensor,
    ) -> ObjectSpaceHeightVolumeEncodingV18:
        """Decode one shared ordered ray field into the sole V18 state."""

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
    ) -> ObjectSpaceHeightVolumeAuxiliaryEncodingV18:
        """Decode once; keep calibrated geometry supervision-only."""

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
        return ObjectSpaceHeightVolumeAuxiliaryEncodingV18(
            latent=encoded.latent,
            nominal_evidence=encoded.nominal_evidence,
            auxiliary_evidence=auxiliary,
            voxel_inputs=encoded.voxel_inputs,
            voxel_visible=encoded.voxel_visible,
            height_volume=encoded.height_volume,
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_with_evidence(patch_tokens).latent


class ObjectSpaceHeightVolumeSemanticDecoderV18(nn.Module):
    """All-channel local semantic decoder producing FREE/OCCUPIED evidence."""

    def __init__(self) -> None:
        super().__init__()
        caller_rng = torch.random.get_rng_state().clone()
        try:
            self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
            self.evidence_head = nn.Conv2d(
                64,
                2,
                kernel_size=1,
                bias=True,
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(VOLUME_INITIALIZATION_SEED_V18)
            for layer in (self.conv1, self.conv2, self.evidence_head):
                nn.init.xavier_uniform_(
                    layer.weight,
                    gain=1.0,
                    generator=generator,
                )
                nn.init.zeros_(layer.bias)
        finally:
            torch.random.set_rng_state(caller_rng)
        self.activation = nn.GELU(approximate="none")

        count = sum(parameter.numel() for parameter in self.parameters())
        if count != OBJECT_SPACE_HEIGHT_VOLUME_SEMANTIC_PARAMETER_COUNT_V18:
            raise RuntimeError("V18 semantic parameter count changed")

    def evidence_logits(
        self,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            latent.ndim != 4
            or tuple(latent.shape[1:]) != (64, *OUTPUT_SHAPE)
        ):
            raise ValueError("V18 semantic latent must have shape (B,64,64,64)")
        trunk = latent + self.conv2(self.activation(self.conv1(latent)))
        axes = self.evidence_head(trunk)
        if not bool(torch.isfinite(axes).all()):
            raise FloatingPointError("V18 semantic evidence is nonfinite")
        return axes[:, 0], axes[:, 1]

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        free, occupied = self.evidence_logits(latent)
        return neutral_disjoint_ternary_log_probabilities_v12(free, occupied)


class GeometryAnchoredSweptProgressSurvivalJointJepaV18(
    GeometryAnchoredSweptProgressSurvivalJointJepaV12
):
    """V14 objectives over one explicit object-space height-volume state."""

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

        self.bev_lift = ObjectSpaceHeightVolumeLiftV18(n320_fit_model)
        self.semantic_head = ObjectSpaceHeightVolumeSemanticDecoderV18()
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_bev_lift = copy.deepcopy(self.bev_lift)
        self.target_hard_sync_count.fill_(1)
        self.ema_update_count.zero_()
        self._freeze_target()
        self._assert_final_target_identity_v18()
        self._assert_parameter_accounting()

    def _assert_final_target_identity_v18(self) -> None:
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
                raise RuntimeError("V18 final target hard-sync identity failed")
        if int(self.target_hard_sync_count.item()) != 1:
            raise RuntimeError("V18 must account exactly one final hard sync")
        if int(self.ema_update_count.item()) != 0:
            raise RuntimeError("V18 must begin before any EMA update")

    def trainable_parameter_groups_v18(self) -> V13TrainableParameterGroups:
        """Return and validate the three exact disjoint V18 route groups."""

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
            shared=select(SHARED_PARAMETER_PREFIXES_V18),
            representation=select(REPRESENTATION_PARAMETER_PREFIXES_V18),
            predictor=select(PREDICTOR_PARAMETER_PREFIXES_V18),
        )
        selected_names = [name for group in groups for name, _ in group]
        if len(selected_names) != len(set(selected_names)):
            raise RuntimeError("V18 trainable parameter groups overlap")
        if set(selected_names) != {name for name, _ in named}:
            raise RuntimeError("V18 trainable parameter groups do not cover the model")
        return groups

    def trainable_parameter_groups_v13(self) -> V13TrainableParameterGroups:
        """Compatibility spelling used by the inherited training lifecycle."""

        return self.trainable_parameter_groups_v18()

    def trainable_parameter_groups_v14(self) -> V13TrainableParameterGroups:
        """Compatibility spelling used by the V14 executor adapter."""

        return self.trainable_parameter_groups_v18()

    def _assert_parameter_accounting(self) -> None:
        groups = self.trainable_parameter_groups_v18()
        counts = tuple(
            sum(parameter.numel() for _, parameter in group) for group in groups
        )
        expected = (
            SHARED_ROUTE_PARAMETER_COUNT_V18,
            REPRESENTATION_GROUP_PARAMETER_COUNT_V18,
            PREDICTOR_GROUP_PARAMETER_COUNT_V18,
        )
        if counts != expected:
            raise RuntimeError(f"V18 online parameter-group counts changed: {counts}")
        if sum(counts) != ONLINE_TRAINABLE_PARAMETER_COUNT_V18:
            raise RuntimeError("V18 total online trainable parameter count changed")
        target_count = sum(
            parameter.numel()
            for module in self.target_modules()
            for parameter in module.parameters()
        )
        if target_count != TARGET_BOTTLENECK_PARAMETER_COUNT_V18:
            raise RuntimeError("V18 target bottleneck parameter count changed")

    def encode_online_with_evidence(
        self,
        rgb: torch.Tensor,
    ) -> ObjectSpaceHeightVolumeEncodingV18:
        self._validate_rgb(rgb, name="online_rgb")
        patch_tokens = self.encoder.forward_tokens(rgb)[:, 1:]
        return self.bev_lift.forward_with_evidence(patch_tokens)

    def encode_online_with_sampling(self, rgb: torch.Tensor) -> None:
        del rgb
        raise RuntimeError(
            "V18 has no collapsed support-sampling schema; "
            "use encode_online_with_evidence"
        )

    @torch.no_grad()
    def encode_target_with_sampling(self, rgb: torch.Tensor) -> None:
        del rgb
        raise RuntimeError(
            "V18 has no collapsed target-sampling schema; use encode_target"
        )

    def encode_online_with_auxiliary_evidence(
        self,
        rgb: torch.Tensor,
        *,
        camera_origin_body_m: torch.Tensor,
        camera_basis_body_fru: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
    ) -> ObjectSpaceHeightVolumeAuxiliaryEncodingV18:
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
    ) -> ObjectSpaceHeightVolumeAuxiliaryEncodingV18:
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
        valid = self.bev_lift.cell_valid_mask[None].expand(
            latent.shape[0], -1, -1
        )
        free = torch.where(valid, free, torch.full_like(free, -20.0))
        occupied = torch.where(valid, occupied, torch.full_like(occupied, -20.0))
        logits = neutral_disjoint_ternary_log_probabilities_v12(free, occupied)
        invalid_logits = logits.new_tensor((0.0, -20.0, -20.0))[
            None, :, None, None
        ]
        return torch.where(valid[:, None], logits, invalid_logits)


ObjectSpaceHeightVolumeJointJepaV18 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV18
)
GeometryAnchoredSweptProgressSurvivalJointJepaV18Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV18
)
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_VOCABULARY_V1",
    "CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13",
    "FLATTENED_VOLUME_CHANNEL_COUNT_V18",
    "FREE_CLASS_V1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1Config",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV18",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV18Config",
    "HEIGHT_CENTRES_M_V18",
    "HEIGHT_COUNT_V18",
    "HEIGHT_NORMALIZATION_CENTRE_M_V18",
    "HEIGHT_NORMALIZATION_HALF_RANGE_M_V18",
    "MIGRATED_EVIDENCE_STATE_COUNT_V14",
    "OBJECT_SPACE_HEIGHT_VOLUME_PARAMETER_COUNT_V18",
    "OBJECT_SPACE_HEIGHT_VOLUME_SEMANTIC_PARAMETER_COUNT_V18",
    "OCCUPIED_CLASS_V1",
    "OFFSET_NORMALIZATION_M_V18",
    "ONLINE_TRAINABLE_PARAMETER_COUNT_V13",
    "ONLINE_TRAINABLE_PARAMETER_COUNT_V18",
    "ObjectSpaceHeightVolumeAuxiliaryEncodingV18",
    "ObjectSpaceHeightVolumeEncodingV18",
    "ObjectSpaceHeightVolumeGeometryV18",
    "ObjectSpaceHeightVolumeJointJepaV18",
    "ObjectSpaceHeightVolumeLiftV18",
    "ObjectSpaceHeightVolumeResidualBlockV18",
    "ObjectSpaceHeightVolumeSemanticDecoderV18",
    "PREDICTOR_GROUP_PARAMETER_COUNT_V13",
    "PREDICTOR_GROUP_PARAMETER_COUNT_V18",
    "PREDICTOR_PARAMETER_PREFIXES_V13",
    "PREDICTOR_PARAMETER_PREFIXES_V18",
    "PROJECTION_INITIALIZATION_SEED_V13",
    "PROJECTION_INITIALIZATION_SEED_V18",
    "REPRESENTATION_GROUP_PARAMETER_COUNT_V13",
    "REPRESENTATION_GROUP_PARAMETER_COUNT_V18",
    "REPRESENTATION_PARAMETER_PREFIXES_V13",
    "REPRESENTATION_PARAMETER_PREFIXES_V18",
    "SHARED_PARAMETER_PREFIXES_V13",
    "SHARED_PARAMETER_PREFIXES_V18",
    "SHARED_ROUTE_PARAMETER_COUNT_V13",
    "SHARED_ROUTE_PARAMETER_COUNT_V18",
    "SWEEP_PROGRESS_BIN_COUNT_V1",
    "TARGET_BOTTLENECK_PARAMETER_COUNT_V13",
    "TARGET_BOTTLENECK_PARAMETER_COUNT_V18",
    "TARGET_PARAMETER_PREFIXES_V13",
    "TARGET_PARAMETER_PREFIXES_V18",
    "UNKNOWN_CLASS_V1",
    "VOXEL_INPUT_CHANNEL_COUNT_V18",
    "VOLUME_CHANNEL_COUNT_V18",
    "VOLUME_INITIALIZATION_SEED_V18",
    "V13TrainableParameterGroups",
    "final_class_macro_nll_per_row",
    "final_class_macro_nll_per_row_v1",
    "flatten_height_major_volume_v18",
    "latent_energy_per_row",
    "latent_energy_per_row_v1",
    "neutral_disjoint_ternary_log_probabilities_v12",
    "object_space_voxel_geometry_v18",
    "ordered_ray_volume_source_v18",
    "unflatten_height_major_volume_v18",
]
