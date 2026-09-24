"""Causal motion-conditioned alignment before the frozen temporal residual.

This additive successor preserves the complete multiresolution temporal V1
encoder, evidence decoder, and token-difference residual.  It inserts one
stateless learned dense warp that samples the previous 16 x 16 token map in
current-grid coordinates.  Callers provide only deployment-style camera bases,
the prior issued-command nominal SE(2), and a reset-safe history-valid mask.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.observable_camera_ray_evidence_v4 import (
    ENCODER_DIM,
    ObservableCameraRayEvidenceV4Model,
    TOKEN_SIDE,
)
from lewm.models.shared_observable_camera_ray_jepa_v5 import (
    SharedObservableCameraRayJepaV5Config,
    SharedOnlineFrameV5,
    tensor_state_dict_sha256,
)
from lewm.models.shared_observable_camera_ray_jepa_v5_multires_temporal_v1 import (
    BASE_INITIALIZATION_SEED,
    DECODER_INITIALIZATION_SEED,
    EXPECTED_ENCODER_PARAMETER_COUNT,
    EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT,
    EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT as TEMPORAL_HEAD_PARAMETER_COUNT,
    EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT as TEMPORAL_HEAD_TENSOR_COUNT,
    EXPECTED_TEMPORAL_PARAMETER_COUNT,
    EXPECTED_TEMPORAL_PARAMETER_TENSOR_COUNT,
    HISTORY_LAG_SECONDS,
    HISTORY_LAG_TICKS,
    N320_CHECKPOINT_CONTENT_SHA256,
    N320_CHECKPOINT_FILE_SHA256,
    SharedObservableCameraRayJepaV5MultiresTemporalV1,
    TEMPORAL_INITIALIZATION_SEED,
    TEMPORAL_WIDTH,
    TemporalMultiresInitializationReceiptV1,
    VisualTokenDifferenceResidualV1,
)


MODEL_FAMILY = (
    "shared_observable_camera_ray_jepa_v5_multires_motion_alignment_v1"
)
ARCHITECTURE_SCHEMA = (
    "lewm_go2_shared_jepa_v5_multires_motion_alignment_v1_architecture"
)
INITIALIZATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_multires_motion_alignment_v1_initialization"
)
ALIGNMENT_INITIALIZATION_SEED = 20260726
MOTION_CONDITION_DIM = 5
ALIGNMENT_WIDTH = 32
MAXIMUM_OFFSET_TOKENS = 2.0
BASIS_ORTHONORMAL_ATOL = 5e-5

EXPECTED_ALIGNMENT_PARAMETER_COUNT = 12_832
EXPECTED_ALIGNMENT_PARAMETER_TENSOR_COUNT = 4
EXPECTED_POST_ENCODER_PARAMETER_COUNT = 15_992
EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT = 368_681
EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT = 35
EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT = 3_116_201
EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT = 113


def _parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def _parameter_tensor_count(module: nn.Module) -> int:
    return sum(1 for _parameter in module.parameters())


def _validate_basis_batch(
    value: torch.Tensor,
    *,
    name: str,
    batch: int,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or tuple(value.shape) != (batch, 3, 3)
        or not value.is_floating_point()
    ):
        raise ValueError(f"{name} must be floating with shape (B,3,3)")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains a nonfinite value")

    value64 = value.to(dtype=torch.float64)
    forward = value64[:, 0]
    right = value64[:, 1]
    up = value64[:, 2]
    reconstructed = torch.stack((forward, -right, up), dim=2)
    identity = torch.eye(
        3,
        dtype=torch.float64,
        device=value.device,
    )[None].expand(batch, -1, -1)
    gram = reconstructed.transpose(1, 2) @ reconstructed
    if not torch.allclose(
        gram,
        identity,
        rtol=0.0,
        atol=BASIS_ORTHONORMAL_ATOL,
    ):
        raise ValueError(f"{name} is not orthonormal")
    if not torch.allclose(
        torch.cross(right, forward, dim=1),
        up,
        rtol=0.0,
        atol=BASIS_ORTHONORMAL_ATOL,
    ):
        raise ValueError(f"{name} violates FRU handedness")
    determinant = torch.linalg.det(reconstructed)
    if not torch.allclose(
        determinant,
        torch.ones_like(determinant),
        rtol=0.0,
        atol=BASIS_ORTHONORMAL_ATOL,
    ):
        raise ValueError(f"{name} reconstructed determinant must equal +1")
    return value


def _basis_roll_pitch(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    forward = value[:, 0]
    right = value[:, 1]
    up = value[:, 2]
    pitch = torch.atan2(
        -forward[:, 2],
        torch.hypot(forward[:, 0], forward[:, 1]),
    )
    roll = torch.atan2(-right[:, 2], up[:, 2])
    return roll, pitch


def _wrapped_difference(
    current: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    difference = current - previous
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def build_causal_motion_condition_v1(
    previous_camera_basis_body_fru: torch.Tensor,
    current_camera_basis_body_fru: torch.Tensor,
    nominal_delta_current_frame: torch.Tensor,
    history_valid: torch.Tensor,
) -> torch.Tensor:
    """Return ``[nominal F,L,yaw, delta roll,pitch]`` with exact cold zeros.

    Stored camera bases have rows ``[forward, right, up]`` in the yaw-aligned
    body frame.  ``right == cross(forward, up)``; therefore the proper rotation
    columns are ``[forward, -right, up]``.
    """

    if (
        not isinstance(previous_camera_basis_body_fru, torch.Tensor)
        or previous_camera_basis_body_fru.ndim != 3
    ):
        raise ValueError(
            "previous_camera_basis_body_fru must be floating with shape (B,3,3)"
        )
    batch = int(previous_camera_basis_body_fru.shape[0])
    previous = _validate_basis_batch(
        previous_camera_basis_body_fru,
        name="previous_camera_basis_body_fru",
        batch=batch,
    )
    current = _validate_basis_batch(
        current_camera_basis_body_fru,
        name="current_camera_basis_body_fru",
        batch=batch,
    )
    if previous.device != current.device or previous.dtype != current.dtype:
        raise ValueError("previous/current camera bases must share device and dtype")
    if (
        not isinstance(nominal_delta_current_frame, torch.Tensor)
        or nominal_delta_current_frame.ndim != 2
        or tuple(nominal_delta_current_frame.shape) != (batch, 3)
        or not nominal_delta_current_frame.is_floating_point()
    ):
        raise ValueError(
            "nominal_delta_current_frame must be floating with shape (B,3)"
        )
    if (
        nominal_delta_current_frame.device != current.device
        or nominal_delta_current_frame.dtype != current.dtype
    ):
        raise ValueError("nominal delta and camera bases must share device and dtype")
    if not bool(torch.isfinite(nominal_delta_current_frame).all()):
        raise FloatingPointError("nominal_delta_current_frame contains a nonfinite value")
    if (
        not isinstance(history_valid, torch.Tensor)
        or history_valid.dtype is not torch.bool
        or tuple(history_valid.shape) != (batch,)
        or history_valid.device != current.device
    ):
        raise ValueError("history_valid must be a device-matched bool tensor (B,)")

    previous_roll, previous_pitch = _basis_roll_pitch(previous)
    current_roll, current_pitch = _basis_roll_pitch(current)
    condition = torch.cat(
        (
            nominal_delta_current_frame,
            _wrapped_difference(current_roll, previous_roll)[:, None],
            _wrapped_difference(current_pitch, previous_pitch)[:, None],
        ),
        dim=1,
    )
    condition = torch.where(
        history_valid[:, None],
        condition,
        torch.zeros_like(condition),
    )
    if (
        tuple(condition.shape) != (batch, MOTION_CONDITION_DIM)
        or not bool(torch.isfinite(condition).all())
    ):
        raise FloatingPointError("causal motion condition is malformed or nonfinite")
    if bool((condition[~history_valid] != 0).any()):
        raise RuntimeError("cold causal motion condition is not exact zero")
    return condition


class MotionConditionedTokenAlignmentV1(nn.Module):
    """Content-aware previous-token sampler conditioned by five causal values."""

    def __init__(self) -> None:
        super().__init__()
        caller_rng = torch.random.get_rng_state()
        try:
            self.input_projection = nn.Conv2d(
                2 * ENCODER_DIM + MOTION_CONDITION_DIM,
                ALIGNMENT_WIDTH,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.activation = nn.GELU(approximate="none")
            self.spatial_projection = nn.Conv2d(
                ALIGNMENT_WIDTH,
                ALIGNMENT_WIDTH,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=ALIGNMENT_WIDTH,
                bias=False,
                padding_mode="zeros",
            )
            self.offset_projection = nn.Conv2d(
                ALIGNMENT_WIDTH,
                2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(ALIGNMENT_INITIALIZATION_SEED)
            with torch.no_grad():
                nn.init.xavier_uniform_(
                    self.input_projection.weight,
                    gain=1.0,
                    generator=generator,
                )
                nn.init.zeros_(self.input_projection.bias)
                nn.init.xavier_uniform_(
                    self.spatial_projection.weight,
                    gain=1.0,
                    generator=generator,
                )
                nn.init.zeros_(self.offset_projection.weight)
        finally:
            torch.random.set_rng_state(caller_rng)

        coordinates = torch.linspace(-1.0, 1.0, TOKEN_SIDE, dtype=torch.float32)
        rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
        # grid_sample consumes [x/column, y/row] in its final dimension.
        identity_grid = torch.stack((columns, rows), dim=-1)[None]
        self.register_buffer("identity_grid_xy", identity_grid, persistent=False)

        if (
            _parameter_count(self) != EXPECTED_ALIGNMENT_PARAMETER_COUNT
            or _parameter_tensor_count(self)
            != EXPECTED_ALIGNMENT_PARAMETER_TENSOR_COUNT
        ):
            raise RuntimeError("motion-alignment capacity contract changed")

    @staticmethod
    def _validate_condition(
        condition: torch.Tensor,
        *,
        tokens: torch.Tensor,
    ) -> None:
        if (
            not isinstance(condition, torch.Tensor)
            or condition.ndim != 2
            or tuple(condition.shape)
            != (tokens.shape[0], MOTION_CONDITION_DIM)
            or not condition.is_floating_point()
            or condition.device != tokens.device
            or condition.dtype != tokens.dtype
        ):
            raise ValueError(
                "motion condition must match tokens with floating shape (B,5)"
            )
        if not bool(torch.isfinite(condition).all()):
            raise FloatingPointError("motion condition contains a nonfinite value")

    def predict_offset_tokens(
        self,
        previous_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        condition: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        VisualTokenDifferenceResidualV1._validate_tokens(
            previous_tokens,
            current_tokens,
            history_valid,
        )
        self._validate_condition(condition, tokens=current_tokens)
        batch = current_tokens.shape[0]
        previous_map = previous_tokens.transpose(1, 2).reshape(
            batch,
            ENCODER_DIM,
            TOKEN_SIDE,
            TOKEN_SIDE,
        )
        current_map = current_tokens.transpose(1, 2).reshape(
            batch,
            ENCODER_DIM,
            TOKEN_SIDE,
            TOKEN_SIDE,
        )
        condition_map = condition[:, :, None, None].expand(
            -1,
            -1,
            TOKEN_SIDE,
            TOKEN_SIDE,
        )
        hidden = self.activation(
            self.input_projection(
                torch.cat((previous_map, current_map, condition_map), dim=1)
            )
        )
        hidden = self.activation(self.spatial_projection(hidden))
        offset = MAXIMUM_OFFSET_TOKENS * torch.tanh(
            self.offset_projection(hidden)
        )
        offset = torch.where(
            history_valid[:, None, None, None],
            offset,
            torch.zeros_like(offset),
        )
        if (
            tuple(offset.shape) != (batch, 2, TOKEN_SIDE, TOKEN_SIDE)
            or not bool(torch.isfinite(offset).all())
            or bool((offset.abs() > MAXIMUM_OFFSET_TOKENS).any())
        ):
            raise FloatingPointError("motion alignment produced invalid offsets")
        return offset

    def warp_previous_tokens(
        self,
        previous_tokens: torch.Tensor,
        offset_tokens_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Sample previous tokens at ``identity + [x/column, y/row]`` offsets."""

        if (
            not isinstance(previous_tokens, torch.Tensor)
            or previous_tokens.ndim != 3
            or tuple(previous_tokens.shape[1:])
            != (TOKEN_SIDE * TOKEN_SIDE, ENCODER_DIM)
            or not previous_tokens.is_floating_point()
        ):
            raise ValueError(
                "previous_tokens must be floating with shape (B,256,192)"
            )
        batch = previous_tokens.shape[0]
        if (
            not isinstance(offset_tokens_xy, torch.Tensor)
            or offset_tokens_xy.ndim != 4
            or tuple(offset_tokens_xy.shape)
            != (batch, 2, TOKEN_SIDE, TOKEN_SIDE)
            or not offset_tokens_xy.is_floating_point()
            or offset_tokens_xy.device != previous_tokens.device
            or offset_tokens_xy.dtype != previous_tokens.dtype
        ):
            raise ValueError(
                "offset_tokens_xy must match tokens with shape (B,2,16,16)"
            )
        if not bool(
            torch.isfinite(previous_tokens).all()
            and torch.isfinite(offset_tokens_xy).all()
        ):
            raise FloatingPointError("warp inputs contain a nonfinite value")
        previous_map = previous_tokens.transpose(1, 2).reshape(
            batch,
            ENCODER_DIM,
            TOKEN_SIDE,
            TOKEN_SIDE,
        )
        normalized_offset_xy = (
            offset_tokens_xy.permute(0, 2, 3, 1)
            * (2.0 / float(TOKEN_SIDE - 1))
        )
        source_grid_xy = self.identity_grid_xy.to(
            device=previous_tokens.device,
            dtype=previous_tokens.dtype,
        ) + normalized_offset_xy
        aligned_map = F.grid_sample(
            previous_map,
            source_grid_xy,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        aligned = aligned_map.flatten(2).transpose(1, 2)
        if not bool(torch.isfinite(aligned).all()):
            raise FloatingPointError("motion-aligned tokens contain a nonfinite value")
        return aligned

    def forward(
        self,
        previous_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        condition: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        offset = self.predict_offset_tokens(
            previous_tokens,
            current_tokens,
            condition,
            history_valid,
        )
        aligned = self.warp_previous_tokens(previous_tokens, offset)
        # Cold rows bypass learned sampling exactly, even after the block learns.
        return torch.where(
            history_valid[:, None, None],
            aligned,
            previous_tokens,
        )


@dataclass(frozen=True)
class MotionAlignmentMultiresInitializationReceiptV1(
    TemporalMultiresInitializationReceiptV1
):
    alignment_initialization_seed: int
    alignment_state_sha256: str
    copied_alignment_entry_count: int
    alignment_offset_projection_exact_zero: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["copied_state_keys"] = list(self.copied_state_keys)
        return value


class SharedObservableCameraRayJepaV5MultiresMotionAlignmentV1(
    SharedObservableCameraRayJepaV5MultiresTemporalV1
):
    """Multires temporal V1 plus one stateless causal learned token warp."""

    model_family = MODEL_FAMILY

    def __init__(
        self,
        config: SharedObservableCameraRayJepaV5Config | None = None,
    ) -> None:
        super().__init__(config=config)
        self.evidence_head.motion_alignment = MotionConditionedTokenAlignmentV1()

        if (
            _parameter_count(self.evidence_head)
            != EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT
            or _parameter_tensor_count(self.evidence_head)
            != EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT
            or TEMPORAL_HEAD_PARAMETER_COUNT
            + EXPECTED_ALIGNMENT_PARAMETER_COUNT
            != EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT
            or TEMPORAL_HEAD_TENSOR_COUNT
            + EXPECTED_ALIGNMENT_PARAMETER_TENSOR_COUNT
            != EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT
            or EXPECTED_TEMPORAL_PARAMETER_COUNT
            + EXPECTED_ALIGNMENT_PARAMETER_COUNT
            != EXPECTED_POST_ENCODER_PARAMETER_COUNT
        ):
            raise RuntimeError("motion-aligned evidence-head capacity contract changed")

        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                name.startswith(("encoder.", "evidence_head."))
            )
        selected = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if self.model_config.encoder_depth == 6 and (
            _parameter_count(self.encoder) != EXPECTED_ENCODER_PARAMETER_COUNT
            or _parameter_tensor_count(self.encoder)
            != EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT
            or sum(int(parameter.numel()) for parameter in selected)
            != EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT
            or len(selected) != EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT
        ):
            raise RuntimeError("motion-alignment trainable partition contract changed")

    @property
    def motion_alignment(self) -> MotionConditionedTokenAlignmentV1:
        module = self.evidence_head.motion_alignment
        if not isinstance(module, MotionConditionedTokenAlignmentV1):
            raise RuntimeError("motion-alignment module identity changed")
        return module

    @staticmethod
    def build_causal_motion_condition(
        previous_camera_basis_body_fru: torch.Tensor,
        current_camera_basis_body_fru: torch.Tensor,
        nominal_delta_current_frame: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        return build_causal_motion_condition_v1(
            previous_camera_basis_body_fru,
            current_camera_basis_body_fru,
            nominal_delta_current_frame,
            history_valid,
        )

    def _migrate_from_n320_fit_model(
        self,
        fit_model: ObservableCameraRayEvidenceV4Model,
        *,
        n320_checkpoint_file_sha256: str,
        n320_checkpoint_content_sha256: str,
    ) -> MotionAlignmentMultiresInitializationReceiptV1:
        alignment_before = tensor_state_dict_sha256(
            self.motion_alignment.state_dict()
        )
        temporal = super()._migrate_from_n320_fit_model(
            fit_model,
            n320_checkpoint_file_sha256=n320_checkpoint_file_sha256,
            n320_checkpoint_content_sha256=n320_checkpoint_content_sha256,
        )
        alignment_after = tensor_state_dict_sha256(
            self.motion_alignment.state_dict()
        )
        if (
            alignment_before != alignment_after
            or any("motion_alignment" in name for name in temporal.copied_state_keys)
            or torch.count_nonzero(
                self.motion_alignment.offset_projection.weight
            ).item()
            != 0
        ):
            raise RuntimeError("N320 migration changed motion-alignment state")
        inherited = asdict(temporal)
        inherited["schema"] = INITIALIZATION_SCHEMA
        inherited["model_family"] = MODEL_FAMILY
        return MotionAlignmentMultiresInitializationReceiptV1(
            **inherited,
            alignment_initialization_seed=ALIGNMENT_INITIALIZATION_SEED,
            alignment_state_sha256=alignment_after,
            copied_alignment_entry_count=0,
            alignment_offset_projection_exact_zero=True,
        )

    def _fuse_motion_aligned_tokens(
        self,
        previous_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        previous_camera_basis_body_fru: torch.Tensor,
        current_camera_basis_body_fru: torch.Tensor,
        nominal_delta_current_frame: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.build_causal_motion_condition(
            previous_camera_basis_body_fru,
            current_camera_basis_body_fru,
            nominal_delta_current_frame,
            history_valid,
        )
        if (
            condition.device != current_tokens.device
            or condition.dtype != current_tokens.dtype
        ):
            raise ValueError("motion condition and visual tokens must share device and dtype")
        aligned_previous = self.motion_alignment(
            previous_tokens,
            current_tokens,
            condition,
            history_valid,
        )
        return self.temporal_residual(
            aligned_previous,
            current_tokens,
            history_valid,
        )

    def forward_temporal_frame(
        self,
        previous_image: torch.Tensor,
        current_image: torch.Tensor,
        previous_camera_basis_body_fru: torch.Tensor,
        target_camera_origin_body_m: torch.Tensor,
        target_camera_basis_body_fru: torch.Tensor,
        target_ground_plane_z_body_m: torch.Tensor,
        nominal_delta_current_frame: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> SharedOnlineFrameV5:
        """Evaluate a mixed warm/cold batch without retaining model state."""

        previous_tokens, current_tokens = self._encode_pair(
            previous_image,
            current_image,
        )
        fused_tokens = self._fuse_motion_aligned_tokens(
            previous_tokens,
            current_tokens,
            previous_camera_basis_body_fru,
            target_camera_basis_body_fru,
            nominal_delta_current_frame,
            history_valid,
        )
        current_bev = self.bev_decoder(current_tokens)
        evidence = self.evidence_head(
            fused_tokens,
            target_camera_origin_body_m,
            target_camera_basis_body_fru,
            target_ground_plane_z_body_m,
        )
        return SharedOnlineFrameV5(
            patch_tokens=current_tokens,
            bev=current_bev,
            evidence=evidence,
            camera_origin_body_m=target_camera_origin_body_m,
            camera_basis_body_fru=target_camera_basis_body_fru,
            ground_plane_z_body_m=target_ground_plane_z_body_m,
        )

    def forward_camera_pair(
        self,
        previous_image: torch.Tensor,
        current_image: torch.Tensor,
        previous_camera_origin_body_m: torch.Tensor,
        previous_camera_basis_body_fru: torch.Tensor,
        previous_ground_plane_z_body_m: torch.Tensor,
        current_camera_origin_body_m: torch.Tensor,
        current_camera_basis_body_fru: torch.Tensor,
        current_ground_plane_z_body_m: torch.Tensor,
        *,
        nominal_delta_current_frame: torch.Tensor,
    ) -> tuple[SharedOnlineFrameV5, SharedOnlineFrameV5]:
        """Return one cold previous frame and one causally warm current frame."""

        previous_tokens, current_tokens = self._encode_pair(
            previous_image,
            current_image,
        )
        previous_bev = self.bev_decoder(previous_tokens)
        current_bev = self.bev_decoder(current_tokens)
        previous_frame = self._frame_from_online(
            previous_tokens,
            previous_bev,
            previous_camera_origin_body_m,
            previous_camera_basis_body_fru,
            previous_ground_plane_z_body_m,
        )
        history_valid = torch.ones(
            (current_tokens.shape[0],),
            dtype=torch.bool,
            device=current_tokens.device,
        )
        fused_tokens = self._fuse_motion_aligned_tokens(
            previous_tokens,
            current_tokens,
            previous_camera_basis_body_fru,
            current_camera_basis_body_fru,
            nominal_delta_current_frame,
            history_valid,
        )
        current_evidence = self.evidence_head(
            fused_tokens,
            current_camera_origin_body_m,
            current_camera_basis_body_fru,
            current_ground_plane_z_body_m,
        )
        current_frame = SharedOnlineFrameV5(
            patch_tokens=current_tokens,
            bev=current_bev,
            evidence=current_evidence,
            camera_origin_body_m=current_camera_origin_body_m,
            camera_basis_body_fru=current_camera_basis_body_fru,
            ground_plane_z_body_m=current_ground_plane_z_body_m,
        )
        return previous_frame, current_frame


def motion_alignment_architecture_contract_v1() -> dict[str, Any]:
    return {
        "schema": ARCHITECTURE_SCHEMA,
        "model_family": MODEL_FAMILY,
        "predecessor_model_family":
            "shared_observable_camera_ray_jepa_v5_multires_temporal_v1",
        "scientific_delta":
            "causal_motion_conditioned_dense_previous_token_alignment",
        "history": {
            "lag_seconds": HISTORY_LAG_SECONDS,
            "lag_ticks": HISTORY_LAG_TICKS,
            "caller_supplied_history_valid": True,
            "model_owned_history_buffer": False,
            "cold_condition_exact_zero": True,
        },
        "condition": {
            "values": [
                "nominal_forward_m",
                "nominal_left_m",
                "nominal_yaw_rad",
                "relative_roll_rad",
                "relative_pitch_rad",
            ],
            "primitive_id_input": False,
            "realized_se2_input": False,
            "basis_orthonormal_rtol": 0.0,
            "basis_orthonormal_atol": BASIS_ORTHONORMAL_ATOL,
        },
        "alignment": {
            "input_channels": 2 * ENCODER_DIM + MOTION_CONDITION_DIM,
            "width": ALIGNMENT_WIDTH,
            "maximum_offset_tokens": MAXIMUM_OFFSET_TOKENS,
            "source_grid": "identity_plus_offset_xy",
            "offset_channel_order": ["x_column", "y_row"],
            "normalized_offset_scale": 2.0 / float(TOKEN_SIDE - 1),
            "grid_sample_mode": "bilinear",
            "grid_sample_padding_mode": "border",
            "grid_sample_align_corners": True,
            "initialization_seed": ALIGNMENT_INITIALIZATION_SEED,
            "offset_projection_exact_zero_at_initialization": True,
            "parameter_count": EXPECTED_ALIGNMENT_PARAMETER_COUNT,
            "parameter_tensor_count": EXPECTED_ALIGNMENT_PARAMETER_TENSOR_COUNT,
        },
        "retained_temporal": {
            "width": TEMPORAL_WIDTH,
            "initialization_seed": TEMPORAL_INITIALIZATION_SEED,
            "parameter_count": EXPECTED_TEMPORAL_PARAMETER_COUNT,
            "parameter_tensor_count": EXPECTED_TEMPORAL_PARAMETER_TENSOR_COUNT,
        },
        "evidence_head": {
            "parameter_count": EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT,
            "parameter_tensor_count":
                EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT,
        },
        "trainable": {
            "encoder_parameter_count": EXPECTED_ENCODER_PARAMETER_COUNT,
            "encoder_parameter_tensor_count":
                EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT,
            "total_parameter_count": EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT,
            "total_parameter_tensor_count":
                EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT,
        },
        "separate_online_encoder_calls": True,
        "target_encoder_calls": 0,
        "jepa_objective_count": 0,
    }


__all__ = [
    "ALIGNMENT_INITIALIZATION_SEED",
    "ALIGNMENT_WIDTH",
    "ARCHITECTURE_SCHEMA",
    "BASIS_ORTHONORMAL_ATOL",
    "EXPECTED_ALIGNMENT_PARAMETER_COUNT",
    "EXPECTED_ALIGNMENT_PARAMETER_TENSOR_COUNT",
    "EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT",
    "EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT",
    "EXPECTED_POST_ENCODER_PARAMETER_COUNT",
    "EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT",
    "EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT",
    "INITIALIZATION_SCHEMA",
    "MAXIMUM_OFFSET_TOKENS",
    "MODEL_FAMILY",
    "MOTION_CONDITION_DIM",
    "MotionAlignmentMultiresInitializationReceiptV1",
    "MotionConditionedTokenAlignmentV1",
    "N320_CHECKPOINT_CONTENT_SHA256",
    "N320_CHECKPOINT_FILE_SHA256",
    "SharedObservableCameraRayJepaV5MultiresMotionAlignmentV1",
    "build_causal_motion_condition_v1",
    "motion_alignment_architecture_contract_v1",
]
