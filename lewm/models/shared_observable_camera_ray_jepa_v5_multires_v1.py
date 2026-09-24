"""Progressive multiresolution evidence decoder for the Shared-V5 scaffold.

This module changes only the token-to-dense evidence decoder.  It deliberately
inherits the established Shared-V5 frame, loss, geometry, and deployment
interfaces so the bounded perception probe changes one scientific mechanism.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.observable_camera_ray_evidence_v4 import (
    DENSE_FEATURE_DIM,
    ENCODER_DIM,
    IMAGE_SIZE,
    ObservableCameraRayEvidenceV4Model,
    TOKEN_SIDE,
)
from lewm.models.shared_observable_camera_ray_jepa_v5 import (
    ObservableCameraRayEvidenceV4Head,
    SharedObservableCameraRayJepaV5,
    SharedObservableCameraRayJepaV5Config,
    tensor_state_dict_sha256,
)


MODEL_FAMILY = "shared_observable_camera_ray_jepa_v5_multires_v1"
ARCHITECTURE_SCHEMA = "lewm_go2_shared_jepa_v5_multires_v1_architecture"
INITIALIZATION_SCHEMA = "lewm_go2_shared_jepa_v5_multires_v1_initialization"
DECODER_INITIALIZATION_SEED = 20260724
BASE_INITIALIZATION_SEED = 20260712
PREDECESSOR_EVIDENCE_HEAD_PARAMETER_CEILING = 357_993
EXPECTED_DECODER_PARAMETER_COUNT = 345_264
EXPECTED_DECODER_PARAMETER_TENSOR_COUNT = 20
EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT = 352_689
EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT = 26
EXPECTED_ENCODER_PARAMETER_COUNT = 2_747_520
EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT = 78
EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT = 3_100_209
EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT = 104
N320_CHECKPOINT_FILE_SHA256 = (
    "ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0"
)
N320_CHECKPOINT_CONTENT_SHA256 = (
    "9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b"
)


@dataclass(frozen=True)
class MultiresDecoderV1Config:
    """Literal reviewed topology for the one-mechanism successor."""

    input_channels: int = ENCODER_DIM
    stage_channels: tuple[int, ...] = (112, 80, 56, 36, 36)
    stage_sizes: tuple[tuple[int, int], ...] = (
        (16, 16),
        (28, 28),
        (56, 56),
        (112, 112),
        (112, 112),
    )
    group_counts: tuple[int, ...] = (8, 8, 8, 4, 4)
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    convolution_bias: bool = True
    resize_mode: str = "bilinear"
    align_corners: bool = False
    antialias: bool = False
    gelu_approximate: str = "none"
    group_norm_eps: float = 1e-5
    group_norm_affine: bool = True
    initialization_seed: int = DECODER_INITIALIZATION_SEED

    def __post_init__(self) -> None:
        if (
            self.input_channels != 192
            or self.stage_channels != (112, 80, 56, 36, 36)
            or self.stage_sizes
            != ((16, 16), (28, 28), (56, 56), (112, 112), (112, 112))
            or self.group_counts != (8, 8, 8, 4, 4)
            or self.kernel_size != 3
            or self.stride != 1
            or self.padding != 1
            or self.convolution_bias is not True
            or self.resize_mode != "bilinear"
            or self.align_corners is not False
            or self.antialias is not False
            or self.gelu_approximate != "none"
            or self.group_norm_eps != 1e-5
            or self.group_norm_affine is not True
            or self.initialization_seed != 20260724
        ):
            raise ValueError("multires decoder V1 configuration is immutable")
        if any(
            channels % groups
            for channels, groups in zip(
                self.stage_channels, self.group_counts, strict=True
            )
        ):
            raise ValueError("GroupNorm groups must divide their stage channels")


def _parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def _parameter_tensor_count(module: nn.Module) -> int:
    return sum(1 for _parameter in module.parameters())


def _stage(
    input_channels: int,
    output_channels: int,
    groups: int,
    *,
    config: MultiresDecoderV1Config,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            bias=config.convolution_bias,
            padding_mode="zeros",
        ),
        nn.GroupNorm(
            groups,
            output_channels,
            eps=config.group_norm_eps,
            affine=config.group_norm_affine,
        ),
        nn.GELU(approximate=config.gelu_approximate),
    )


class ProgressiveSpatialEvidenceDecoderV1(nn.Module):
    """Map final 16x16 ViT tokens through the frozen five-stage topology."""

    def __init__(
        self, config: MultiresDecoderV1Config | None = None
    ) -> None:
        super().__init__()
        self.config = config or MultiresDecoderV1Config()
        if not isinstance(self.config, MultiresDecoderV1Config):
            raise TypeError("config must be MultiresDecoderV1Config")

        caller_rng = torch.random.get_rng_state()
        try:
            channels = (
                self.config.input_channels,
                *self.config.stage_channels,
            )
            self.stages = nn.ModuleList(
                _stage(
                    channels[index],
                    channels[index + 1],
                    self.config.group_counts[index],
                    config=self.config,
                )
                for index in range(len(self.config.stage_channels))
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.config.initialization_seed)
            with torch.no_grad():
                for module in self.modules():
                    if isinstance(module, nn.Conv2d):
                        nn.init.xavier_uniform_(
                            module.weight,
                            gain=1.0,
                            generator=generator,
                        )
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.GroupNorm):
                        assert module.weight is not None and module.bias is not None
                        nn.init.ones_(module.weight)
                        nn.init.zeros_(module.bias)
        finally:
            torch.random.set_rng_state(caller_rng)

        if (
            _parameter_count(self) != EXPECTED_DECODER_PARAMETER_COUNT
            or _parameter_tensor_count(self)
            != EXPECTED_DECODER_PARAMETER_TENSOR_COUNT
        ):
            raise RuntimeError("multires decoder parameter contract changed")

    def forward_with_shapes(
        self, token_map: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
        if (
            not isinstance(token_map, torch.Tensor)
            or token_map.ndim != 4
            or tuple(token_map.shape[1:])
            != (ENCODER_DIM, TOKEN_SIDE, TOKEN_SIDE)
            or not token_map.is_floating_point()
        ):
            raise ValueError(
                "token_map must have floating shape (B,192,16,16)"
            )
        if not bool(torch.isfinite(token_map).all()):
            raise FloatingPointError("token_map contains a nonfinite value")

        shapes: list[tuple[int, ...]] = []
        value = token_map
        for index, (stage, output_size) in enumerate(
            zip(self.stages, self.config.stage_sizes, strict=True)
        ):
            if index and tuple(value.shape[-2:]) != output_size:
                value = F.interpolate(
                    value,
                    size=output_size,
                    mode=self.config.resize_mode,
                    align_corners=self.config.align_corners,
                    antialias=self.config.antialias,
                )
            value = stage(value)
            if tuple(value.shape[-2:]) != output_size:
                raise RuntimeError("multires stage returned an unexpected shape")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(
                    "multires decoder produced a nonfinite value"
                )
            shapes.append(tuple(value.shape))

        expected = (
            token_map.shape[0],
            DENSE_FEATURE_DIM,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
        if tuple(value.shape) != expected:
            raise RuntimeError("multires decoder returned an unexpected output")
        return value, tuple(shapes)

    def forward(self, token_map: torch.Tensor) -> torch.Tensor:
        value, _shapes = self.forward_with_shapes(token_map)
        return value


class ObservableCameraRayEvidenceMultiresHeadV1(
    ObservableCameraRayEvidenceV4Head
):
    """The unchanged V4 consumers backed by the progressive decoder."""

    def __init__(
        self,
        *,
        source_shape: tuple[int, int],
        pixel_ray_shape: tuple[int, int],
        query_chunk_size: int | None,
    ) -> None:
        super().__init__(
            source_shape=source_shape,
            pixel_ray_shape=pixel_ray_shape,
            query_chunk_size=query_chunk_size,
        )
        self.dense_decoder = ProgressiveSpatialEvidenceDecoderV1()
        if (
            _parameter_count(self) != EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT
            or _parameter_tensor_count(self)
            != EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT
            or _parameter_count(self)
            > PREDECESSOR_EVIDENCE_HEAD_PARAMETER_CEILING
        ):
            raise RuntimeError("multires evidence-head capacity contract changed")

    def migrate_from_fit_model(
        self,
        fit_model: ObservableCameraRayEvidenceV4Model,
    ) -> tuple[str, ...]:
        """Copy only compatible learned consumers; never copy the old decoder."""

        if not isinstance(fit_model, ObservableCameraRayEvidenceV4Model):
            raise TypeError("fit_model must be an ObservableCameraRayEvidenceV4Model")
        if (
            tuple(fit_model.source_shape) != self.source_shape
            or tuple(fit_model.pixel_ray_shape) != self.pixel_ray_shape
        ):
            raise ValueError("fit-model and multires head geometries differ")
        if not torch.equal(
            fit_model.canonical_ground_support_xy_body_m,
            self.canonical_ground_support_xy_body_m,
        ):
            raise ValueError("canonical ground-support buffer changed")

        self.pixel_head.load_state_dict(
            fit_model.pixel_head.state_dict(), strict=True
        )
        self.ground_head.load_state_dict(
            fit_model.ground_head.state_dict(), strict=True
        )
        copied = tuple(
            sorted(
                (
                    *(f"pixel_head.{name}" for name in self.pixel_head.state_dict()),
                    *(f"ground_head.{name}" for name in self.ground_head.state_dict()),
                )
            )
        )
        if len(copied) != 6 or any(name.startswith("dense_decoder.") for name in copied):
            raise RuntimeError("multires migration copy allowlist changed")
        return copied


@dataclass(frozen=True)
class MultiresInitializationReceiptV1:
    schema: str
    model_family: str
    base_initialization_seed: int
    decoder_initialization_seed: int
    initialization_input_role: str
    n320_checkpoint_file_sha256: str
    n320_checkpoint_content_sha256: str
    fit_model_state_sha256: str
    shared_encoder_state_sha256: str
    pixel_head_state_sha256: str
    ground_head_state_sha256: str
    decoder_state_sha256: str
    evidence_head_state_sha256: str
    copied_state_keys: tuple[str, ...]
    copied_state_entry_count: int
    copied_predecessor_dense_decoder_entry_count: int
    canonical_ground_support_exact: bool
    hard_sync_count: int
    caller_cpu_rng_restored: bool
    rejected_adaptation_checkpoint_open_count: int
    torch_version: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["copied_state_keys"] = list(self.copied_state_keys)
        return value


class SharedObservableCameraRayJepaV5MultiresV1(
    SharedObservableCameraRayJepaV5
):
    """Shared-V5 with only its dense evidence decoder replaced."""

    model_family = MODEL_FAMILY

    def __init__(
        self,
        config: SharedObservableCameraRayJepaV5Config | None = None,
    ) -> None:
        super().__init__(config=config)
        config = self.model_config
        self.evidence_head = ObservableCameraRayEvidenceMultiresHeadV1(
            source_shape=config.source_shape,
            pixel_ray_shape=config.pixel_ray_shape,
            query_chunk_size=config.query_chunk_size,
        )
        self._require_encoder_contract()
        if (
            _parameter_count(self.encoder) != EXPECTED_ENCODER_PARAMETER_COUNT
            or _parameter_tensor_count(self.encoder)
            != EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT
        ) and config.encoder_depth == 6:
            raise RuntimeError("multires encoder parameter contract changed")

        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                name.startswith(("encoder.", "evidence_head."))
            )
        selected = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        if config.encoder_depth == 6 and (
            sum(int(parameter.numel()) for parameter in selected)
            != EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT
            or len(selected)
            != EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT
        ):
            raise RuntimeError("multires trainable partition contract changed")
        self._n320_initialization_complete = False

    def migrate_from_fit_model(
        self,
        fit_model: ObservableCameraRayEvidenceV4Model,
    ) -> MultiresInitializationReceiptV1:
        del fit_model
        raise PermissionError(
            "generic fit-model migration is prohibited; use "
            "initialize_from_n320_fit_model"
        )

    def _migrate_from_n320_fit_model(
        self,
        fit_model: ObservableCameraRayEvidenceV4Model,
        *,
        n320_checkpoint_file_sha256: str,
        n320_checkpoint_content_sha256: str,
    ) -> MultiresInitializationReceiptV1:
        if not isinstance(fit_model, ObservableCameraRayEvidenceV4Model):
            raise TypeError("fit_model must be an ObservableCameraRayEvidenceV4Model")
        if (
            n320_checkpoint_file_sha256 != N320_CHECKPOINT_FILE_SHA256
            or n320_checkpoint_content_sha256
            != N320_CHECKPOINT_CONTENT_SHA256
        ):
            raise PermissionError("unbound fit checkpoint is not N320")
        if self._n320_initialization_complete:
            raise PermissionError("N320 initialization is one-shot")
        self.encoder.load_state_dict(fit_model.encoder.state_dict(), strict=True)
        copied_head = self.evidence_head.migrate_from_fit_model(fit_model)
        copied = tuple(
            sorted(
                (
                    *(f"encoder.{name}" for name in self.encoder.state_dict()),
                    *(
                        f"evidence_head.{name}"
                        for name in copied_head
                    ),
                )
            )
        )
        expected_copy_count = len(fit_model.encoder.state_dict()) + 6
        if (
            len(copied) != expected_copy_count
            or (self.model_config.encoder_depth == 6 and len(copied) != 84)
            or not all(
                name.startswith(
                    (
                        "encoder.",
                        "evidence_head.pixel_head.",
                        "evidence_head.ground_head.",
                    )
                )
                for name in copied
            )
            or any("dense_decoder" in name for name in copied)
        ):
            raise RuntimeError("N320 migration copy allowlist changed")
        self.hard_sync_ema_target_from_online()
        self._n320_initialization_complete = True
        return MultiresInitializationReceiptV1(
            schema=INITIALIZATION_SCHEMA,
            model_family=MODEL_FAMILY,
            base_initialization_seed=BASE_INITIALIZATION_SEED,
            decoder_initialization_seed=DECODER_INITIALIZATION_SEED,
            initialization_input_role="n320_fit_initialization_only",
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
            fit_model_state_sha256=tensor_state_dict_sha256(
                fit_model.state_dict()
            ),
            shared_encoder_state_sha256=tensor_state_dict_sha256(
                self.encoder.state_dict()
            ),
            pixel_head_state_sha256=tensor_state_dict_sha256(
                self.evidence_head.pixel_head.state_dict()
            ),
            ground_head_state_sha256=tensor_state_dict_sha256(
                self.evidence_head.ground_head.state_dict()
            ),
            decoder_state_sha256=tensor_state_dict_sha256(
                self.evidence_head.dense_decoder.state_dict()
            ),
            evidence_head_state_sha256=tensor_state_dict_sha256(
                self.evidence_head.state_dict()
            ),
            copied_state_keys=copied,
            copied_state_entry_count=len(copied),
            copied_predecessor_dense_decoder_entry_count=0,
            canonical_ground_support_exact=True,
            hard_sync_count=1,
            caller_cpu_rng_restored=True,
            rejected_adaptation_checkpoint_open_count=0,
            torch_version=str(torch.__version__),
        )

    @classmethod
    def initialize_from_n320_fit_model(
        cls,
        fit_model: ObservableCameraRayEvidenceV4Model,
        *,
        n320_checkpoint_file_sha256: str,
        n320_checkpoint_content_sha256: str,
        config: SharedObservableCameraRayJepaV5Config | None = None,
    ) -> tuple[
        "SharedObservableCameraRayJepaV5MultiresV1",
        MultiresInitializationReceiptV1,
    ]:
        """Construct and one-shot initialize under the reviewed RNG boundary."""

        if not isinstance(fit_model, ObservableCameraRayEvidenceV4Model):
            raise TypeError("fit_model must be an ObservableCameraRayEvidenceV4Model")
        if (
            n320_checkpoint_file_sha256 != N320_CHECKPOINT_FILE_SHA256
            or n320_checkpoint_content_sha256
            != N320_CHECKPOINT_CONTENT_SHA256
        ):
            raise PermissionError("unbound fit checkpoint is not N320")

        caller_rng = torch.random.get_rng_state()
        try:
            torch.manual_seed(BASE_INITIALIZATION_SEED)
            model = cls(config=config)
            receipt = model._migrate_from_n320_fit_model(
                fit_model,
                n320_checkpoint_file_sha256=n320_checkpoint_file_sha256,
                n320_checkpoint_content_sha256=n320_checkpoint_content_sha256,
            )
        finally:
            torch.random.set_rng_state(caller_rng)
        return model, receipt


def multires_architecture_contract_v1() -> dict[str, Any]:
    config = MultiresDecoderV1Config()
    return {
        "schema": ARCHITECTURE_SCHEMA,
        "model_family": MODEL_FAMILY,
        "scientific_delta":
            "single_stride_seven_dense_lift_to_progressive_multiresolution_decoder",
        "decoder": {
            **asdict(config),
            "stage_channels": list(config.stage_channels),
            "stage_sizes": [list(value) for value in config.stage_sizes],
            "group_counts": list(config.group_counts),
            "parameter_count": EXPECTED_DECODER_PARAMETER_COUNT,
            "parameter_tensor_count": EXPECTED_DECODER_PARAMETER_TENSOR_COUNT,
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
        "unchanged_consumers": [
            "pixel_head",
            "ground_head",
            "camera_geometry",
            "ray_depth_ground_output_contract",
            "rasterization",
        ],
        "intermediate_encoder_features_used": False,
    }


__all__ = [
    "ARCHITECTURE_SCHEMA",
    "BASE_INITIALIZATION_SEED",
    "DECODER_INITIALIZATION_SEED",
    "EXPECTED_DECODER_PARAMETER_COUNT",
    "EXPECTED_DECODER_PARAMETER_TENSOR_COUNT",
    "EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT",
    "EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT",
    "EXPECTED_ENCODER_PARAMETER_COUNT",
    "EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT",
    "EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT",
    "EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT",
    "INITIALIZATION_SCHEMA",
    "MODEL_FAMILY",
    "N320_CHECKPOINT_CONTENT_SHA256",
    "N320_CHECKPOINT_FILE_SHA256",
    "MultiresDecoderV1Config",
    "MultiresInitializationReceiptV1",
    "ObservableCameraRayEvidenceMultiresHeadV1",
    "PREDECESSOR_EVIDENCE_HEAD_PARAMETER_CEILING",
    "ProgressiveSpatialEvidenceDecoderV1",
    "SharedObservableCameraRayJepaV5MultiresV1",
    "multires_architecture_contract_v1",
]
