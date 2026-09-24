"""Pure-visual causal temporal fusion before the multires evidence decoder.

This additive successor retains the complete Shared-V5 multiresolution model
and inserts one fixed-width residual over previous/current raw visual tokens.
It owns no history buffer: callers supply a reset- and timing-validated
``history_valid`` mask and retain only raw, never fused, tokens.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

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
from lewm.models.shared_observable_camera_ray_jepa_v5_multires_v1 import (
    BASE_INITIALIZATION_SEED,
    DECODER_INITIALIZATION_SEED,
    EXPECTED_ENCODER_PARAMETER_COUNT,
    EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT,
    EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT as PREDECESSOR_HEAD_PARAMETER_COUNT,
    EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT as PREDECESSOR_HEAD_TENSOR_COUNT,
    N320_CHECKPOINT_CONTENT_SHA256,
    N320_CHECKPOINT_FILE_SHA256,
    PREDECESSOR_EVIDENCE_HEAD_PARAMETER_CEILING,
    SharedObservableCameraRayJepaV5MultiresV1,
)


MODEL_FAMILY = "shared_observable_camera_ray_jepa_v5_multires_temporal_v1"
ARCHITECTURE_SCHEMA = (
    "lewm_go2_shared_jepa_v5_multires_temporal_v1_architecture"
)
INITIALIZATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_multires_temporal_v1_initialization"
)
TEMPORAL_INITIALIZATION_SEED = 20260725
HISTORY_LAG_SECONDS = 0.5
HISTORY_LAG_TICKS = 5
TEMPORAL_WIDTH = 8

EXPECTED_TEMPORAL_PARAMETER_COUNT = 3_160
EXPECTED_TEMPORAL_PARAMETER_TENSOR_COUNT = 5
EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT = 355_849
EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT = 31
EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT = 3_103_369
EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT = 109


def _parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def _parameter_tensor_count(module: nn.Module) -> int:
    return sum(1 for _parameter in module.parameters())


class VisualTokenDifferenceResidualV1(nn.Module):
    """Width-eight learned residual over a reset-safe previous token grid."""

    def __init__(self) -> None:
        super().__init__()
        caller_rng = torch.random.get_rng_state()
        try:
            self.input_projection = nn.Conv2d(
                ENCODER_DIM,
                TEMPORAL_WIDTH,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.normalization = nn.GroupNorm(
                4,
                TEMPORAL_WIDTH,
                eps=1e-5,
                affine=True,
            )
            self.activation = nn.GELU(approximate="none")
            self.spatial_projection = nn.Conv2d(
                TEMPORAL_WIDTH,
                TEMPORAL_WIDTH,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=TEMPORAL_WIDTH,
                bias=False,
                padding_mode="zeros",
            )
            self.output_projection = nn.Conv2d(
                TEMPORAL_WIDTH,
                ENCODER_DIM,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

            generator = torch.Generator(device="cpu")
            generator.manual_seed(TEMPORAL_INITIALIZATION_SEED)
            with torch.no_grad():
                nn.init.xavier_uniform_(
                    self.input_projection.weight,
                    gain=1.0,
                    generator=generator,
                )
                nn.init.ones_(self.normalization.weight)
                nn.init.zeros_(self.normalization.bias)
                nn.init.xavier_uniform_(
                    self.spatial_projection.weight,
                    gain=1.0,
                    generator=generator,
                )
                nn.init.zeros_(self.output_projection.weight)
        finally:
            torch.random.set_rng_state(caller_rng)

        if (
            _parameter_count(self) != EXPECTED_TEMPORAL_PARAMETER_COUNT
            or _parameter_tensor_count(self)
            != EXPECTED_TEMPORAL_PARAMETER_TENSOR_COUNT
        ):
            raise RuntimeError("temporal residual capacity contract changed")

    @staticmethod
    def _validate_tokens(
        previous_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> None:
        expected_tail = (TOKEN_SIDE * TOKEN_SIDE, ENCODER_DIM)
        if (
            not isinstance(previous_tokens, torch.Tensor)
            or not isinstance(current_tokens, torch.Tensor)
            or previous_tokens.ndim != 3
            or current_tokens.ndim != 3
            or tuple(previous_tokens.shape[1:]) != expected_tail
            or tuple(current_tokens.shape[1:]) != expected_tail
            or previous_tokens.shape != current_tokens.shape
            or previous_tokens.shape[0] <= 0
            or not previous_tokens.is_floating_point()
            or not current_tokens.is_floating_point()
        ):
            raise ValueError(
                "previous/current tokens must have matching floating shape "
                "(B,256,192)"
            )
        if (
            previous_tokens.device != current_tokens.device
            or previous_tokens.dtype != current_tokens.dtype
        ):
            raise ValueError("previous/current tokens must share device and dtype")
        if (
            not isinstance(history_valid, torch.Tensor)
            or history_valid.dtype is not torch.bool
            or tuple(history_valid.shape) != (current_tokens.shape[0],)
            or history_valid.device != current_tokens.device
        ):
            raise ValueError("history_valid must be a device-matched bool tensor (B,)")
        if not bool(
            torch.isfinite(previous_tokens).all()
            and torch.isfinite(current_tokens).all()
        ):
            raise FloatingPointError("temporal tokens contain a nonfinite value")

    def forward(
        self,
        previous_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_tokens(previous_tokens, current_tokens, history_valid)
        batch = current_tokens.shape[0]
        delta = (current_tokens - previous_tokens).transpose(1, 2).reshape(
            batch,
            ENCODER_DIM,
            TOKEN_SIDE,
            TOKEN_SIDE,
        )
        hidden = self.activation(
            self.normalization(self.input_projection(delta))
        )
        hidden = self.activation(self.spatial_projection(hidden))
        residual = self.output_projection(hidden).flatten(2).transpose(1, 2)
        fused = torch.where(
            history_valid[:, None, None],
            current_tokens + residual,
            current_tokens,
        )
        if not bool(torch.isfinite(fused).all()):
            raise FloatingPointError("temporal residual produced a nonfinite value")
        return fused


@dataclass(frozen=True)
class TemporalMultiresInitializationReceiptV1:
    schema: str
    model_family: str
    base_initialization_seed: int
    decoder_initialization_seed: int
    temporal_initialization_seed: int
    initialization_input_role: str
    n320_checkpoint_file_sha256: str
    n320_checkpoint_content_sha256: str
    fit_model_state_sha256: str
    shared_encoder_state_sha256: str
    pixel_head_state_sha256: str
    ground_head_state_sha256: str
    decoder_state_sha256: str
    temporal_state_sha256: str
    evidence_head_state_sha256: str
    copied_state_keys: tuple[str, ...]
    copied_state_entry_count: int
    copied_predecessor_dense_decoder_entry_count: int
    copied_temporal_entry_count: int
    temporal_output_projection_exact_zero: bool
    canonical_ground_support_exact: bool
    hard_sync_count: int
    caller_cpu_rng_restored: bool
    rejected_adaptation_checkpoint_open_count: int
    torch_version: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["copied_state_keys"] = list(self.copied_state_keys)
        return value


class SharedObservableCameraRayJepaV5MultiresTemporalV1(
    SharedObservableCameraRayJepaV5MultiresV1
):
    """Multires Shared-V5 with one stateless pure-visual temporal residual."""

    model_family = MODEL_FAMILY

    def __init__(
        self,
        config: SharedObservableCameraRayJepaV5Config | None = None,
    ) -> None:
        super().__init__(config=config)
        self.evidence_head.temporal_residual = VisualTokenDifferenceResidualV1()

        if (
            _parameter_count(self.evidence_head)
            != EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT
            or _parameter_tensor_count(self.evidence_head)
            != EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT
            or _parameter_count(self.evidence_head)
            > PREDECESSOR_EVIDENCE_HEAD_PARAMETER_CEILING
            or PREDECESSOR_HEAD_PARAMETER_COUNT
            + EXPECTED_TEMPORAL_PARAMETER_COUNT
            != EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT
            or PREDECESSOR_HEAD_TENSOR_COUNT
            + EXPECTED_TEMPORAL_PARAMETER_TENSOR_COUNT
            != EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT
        ):
            raise RuntimeError("temporal evidence-head capacity contract changed")

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
            raise RuntimeError("temporal trainable partition contract changed")

    @property
    def temporal_residual(self) -> VisualTokenDifferenceResidualV1:
        module = self.evidence_head.temporal_residual
        if not isinstance(module, VisualTokenDifferenceResidualV1):
            raise RuntimeError("temporal residual module identity changed")
        return module

    def _migrate_from_n320_fit_model(
        self,
        fit_model: ObservableCameraRayEvidenceV4Model,
        *,
        n320_checkpoint_file_sha256: str,
        n320_checkpoint_content_sha256: str,
    ) -> TemporalMultiresInitializationReceiptV1:
        temporal_before = tensor_state_dict_sha256(
            self.temporal_residual.state_dict()
        )
        base = super()._migrate_from_n320_fit_model(
            fit_model,
            n320_checkpoint_file_sha256=n320_checkpoint_file_sha256,
            n320_checkpoint_content_sha256=n320_checkpoint_content_sha256,
        )
        temporal_after = tensor_state_dict_sha256(
            self.temporal_residual.state_dict()
        )
        if (
            temporal_before != temporal_after
            or any("temporal_residual" in name for name in base.copied_state_keys)
            or torch.count_nonzero(
                self.temporal_residual.output_projection.weight
            ).item()
            != 0
        ):
            raise RuntimeError("N320 migration changed temporal state")
        return TemporalMultiresInitializationReceiptV1(
            schema=INITIALIZATION_SCHEMA,
            model_family=MODEL_FAMILY,
            base_initialization_seed=base.base_initialization_seed,
            decoder_initialization_seed=base.decoder_initialization_seed,
            temporal_initialization_seed=TEMPORAL_INITIALIZATION_SEED,
            initialization_input_role=base.initialization_input_role,
            n320_checkpoint_file_sha256=base.n320_checkpoint_file_sha256,
            n320_checkpoint_content_sha256=(
                base.n320_checkpoint_content_sha256
            ),
            fit_model_state_sha256=base.fit_model_state_sha256,
            shared_encoder_state_sha256=base.shared_encoder_state_sha256,
            pixel_head_state_sha256=base.pixel_head_state_sha256,
            ground_head_state_sha256=base.ground_head_state_sha256,
            decoder_state_sha256=base.decoder_state_sha256,
            temporal_state_sha256=temporal_after,
            evidence_head_state_sha256=base.evidence_head_state_sha256,
            copied_state_keys=base.copied_state_keys,
            copied_state_entry_count=base.copied_state_entry_count,
            copied_predecessor_dense_decoder_entry_count=(
                base.copied_predecessor_dense_decoder_entry_count
            ),
            copied_temporal_entry_count=0,
            temporal_output_projection_exact_zero=True,
            canonical_ground_support_exact=base.canonical_ground_support_exact,
            hard_sync_count=base.hard_sync_count,
            caller_cpu_rng_restored=base.caller_cpu_rng_restored,
            rejected_adaptation_checkpoint_open_count=(
                base.rejected_adaptation_checkpoint_open_count
            ),
            torch_version=base.torch_version,
        )

    @staticmethod
    def _validate_image_pair(
        previous_image: torch.Tensor,
        current_image: torch.Tensor,
    ) -> None:
        ObservableCameraRayEvidenceV4Model._validate_image(previous_image)
        ObservableCameraRayEvidenceV4Model._validate_image(current_image)
        if (
            previous_image.shape != current_image.shape
            or previous_image.device != current_image.device
            or previous_image.dtype != current_image.dtype
        ):
            raise ValueError(
                "previous/current images must share shape, device, and dtype"
            )

    def _encode_pair(
        self,
        previous_image: torch.Tensor,
        current_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_encoder_contract()
        self._validate_image_pair(previous_image, current_image)
        previous_tokens = self.encoder.forward_tokens(previous_image)[:, 1:]
        current_tokens = self.encoder.forward_tokens(current_image)[:, 1:]
        expected = (
            previous_image.shape[0],
            TOKEN_SIDE * TOKEN_SIDE,
            ENCODER_DIM,
        )
        if (
            tuple(previous_tokens.shape) != expected
            or tuple(current_tokens.shape) != expected
        ):
            raise RuntimeError("online encoder token contract changed")
        return previous_tokens, current_tokens

    def forward_temporal_frame(
        self,
        previous_image: torch.Tensor,
        current_image: torch.Tensor,
        target_camera_origin_body_m: torch.Tensor,
        target_camera_basis_body_fru: torch.Tensor,
        target_ground_plane_z_body_m: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> SharedOnlineFrameV5:
        """Evaluate a mixed warm/cold batch without retaining model state."""

        previous_tokens, current_tokens = self._encode_pair(
            previous_image,
            current_image,
        )
        fused_tokens = self.temporal_residual(
            previous_tokens,
            current_tokens,
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
        fused_tokens = self.temporal_residual(
            previous_tokens,
            current_tokens,
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

    def forward_training_pair(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise PermissionError(
            "JEPA forward_training_pair bypasses temporal fusion; use "
            "forward_camera_pair for this supervised temporal model"
        )


def temporal_multires_architecture_contract_v1() -> dict[str, Any]:
    return {
        "schema": ARCHITECTURE_SCHEMA,
        "model_family": MODEL_FAMILY,
        "predecessor_model_family":
            "shared_observable_camera_ray_jepa_v5_multires_v1",
        "scientific_delta":
            "pure_visual_fixed_lag_previous_current_token_difference_residual",
        "history": {
            "lag_seconds": HISTORY_LAG_SECONDS,
            "lag_ticks": HISTORY_LAG_TICKS,
            "caller_supplied_history_valid": True,
            "model_owned_history_buffer": False,
            "raw_tokens_only": True,
        },
        "temporal": {
            "input_shape": ["B", TOKEN_SIDE * TOKEN_SIDE, ENCODER_DIM],
            "map_shape": ["B", ENCODER_DIM, TOKEN_SIDE, TOKEN_SIDE],
            "width": TEMPORAL_WIDTH,
            "initialization_seed": TEMPORAL_INITIALIZATION_SEED,
            "parameter_count": EXPECTED_TEMPORAL_PARAMETER_COUNT,
            "parameter_tensor_count":
                EXPECTED_TEMPORAL_PARAMETER_TENSOR_COUNT,
            "context_inputs": [],
            "output_projection_exact_zero_at_initialization": True,
        },
        "evidence_head": {
            "parameter_count": EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT,
            "parameter_tensor_count":
                EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT,
            "predecessor_parameter_ceiling":
                PREDECESSOR_EVIDENCE_HEAD_PARAMETER_CEILING,
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
    "ARCHITECTURE_SCHEMA",
    "BASE_INITIALIZATION_SEED",
    "DECODER_INITIALIZATION_SEED",
    "EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT",
    "EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT",
    "EXPECTED_TEMPORAL_PARAMETER_COUNT",
    "EXPECTED_TEMPORAL_PARAMETER_TENSOR_COUNT",
    "EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT",
    "EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT",
    "HISTORY_LAG_SECONDS",
    "HISTORY_LAG_TICKS",
    "INITIALIZATION_SCHEMA",
    "MODEL_FAMILY",
    "N320_CHECKPOINT_CONTENT_SHA256",
    "N320_CHECKPOINT_FILE_SHA256",
    "SharedObservableCameraRayJepaV5MultiresTemporalV1",
    "TEMPORAL_INITIALIZATION_SEED",
    "TEMPORAL_WIDTH",
    "TemporalMultiresInitializationReceiptV1",
    "VisualTokenDifferenceResidualV1",
    "temporal_multires_architecture_contract_v1",
]
