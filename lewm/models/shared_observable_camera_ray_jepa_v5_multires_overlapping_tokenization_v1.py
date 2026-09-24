"""Overlapping RGB tokenization for the static Shared-V5 multires model.

This additive successor changes only the two visual patch projections.  The
online and EMA-target encoders retain a stride and configured patch size of
seven while using an 11x11, padding-two receptive field.  The expanded
projection is initialized as the exact 7x7 predecessor projection surrounded
by a zero ring.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from lewm.models.observable_camera_ray_evidence_v4 import (
    ENCODER_DIM,
    IMAGE_SIZE,
    ObservableCameraRayEvidenceV4Model,
    TOKEN_SIDE,
)
from lewm.models.shared_observable_camera_ray_jepa_v5 import (
    SharedObservableCameraRayJepaV5Config,
    tensor_state_dict_sha256,
)
from lewm.models.shared_observable_camera_ray_jepa_v5_multires_v1 import (
    BASE_INITIALIZATION_SEED,
    DECODER_INITIALIZATION_SEED,
    EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT,
    EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT,
    N320_CHECKPOINT_CONTENT_SHA256,
    N320_CHECKPOINT_FILE_SHA256,
    SharedObservableCameraRayJepaV5MultiresV1,
    multires_architecture_contract_v1,
)


MODEL_FAMILY = (
    "shared_observable_camera_ray_jepa_v5_multires_"
    "overlapping_tokenization_v1"
)
ARCHITECTURE_SCHEMA = (
    "lewm_go2_shared_jepa_v5_multires_"
    "overlapping_tokenization_v1_architecture"
)
INITIALIZATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_multires_"
    "overlapping_tokenization_v1_initialization"
)
ONE_SCIENCE_DELTA = (
    "overlapping_rgb_patch_tokenization_relative_to_static_multires_v3_only"
)

PATCH_INPUT_CHANNELS = 3
PATCH_OUTPUT_CHANNELS = ENCODER_DIM
PREDECESSOR_PATCH_KERNEL_SIZE = (7, 7)
PATCH_KERNEL_SIZE = (11, 11)
PATCH_STRIDE = (7, 7)
PATCH_PADDING = (2, 2)
PATCH_DILATION = (1, 1)
PATCH_GROUPS = 1
PATCH_BIAS = True
PATCH_PADDING_MODE = "zeros"
CENTER_COPY_SLICE = (2, 9, 2, 9)

EXPECTED_CENTRAL_WEIGHT_SCALAR_COUNT = 28_224
EXPECTED_OUTER_RING_SCALAR_COUNT = 41_472
EXPECTED_PATCH_BIAS_SCALAR_COUNT = 192
EXPECTED_PATCH_WEIGHT_PARAMETER_COUNT = 69_696
EXPECTED_ENCODER_PARAMETER_COUNT = 2_788_992
EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT = 78
EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT = 3_141_681
EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT = 104
EXPECTED_COMPLETE_MODEL_PARAMETER_COUNT = 7_049_460
EXPECTED_COMPLETE_MODEL_PARAMETER_TENSOR_COUNT = 232


def _parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def _parameter_tensor_count(module: nn.Module) -> int:
    return sum(1 for _parameter in module.parameters())


def _require_patch_projection(
    projection: nn.Conv2d,
    *,
    kernel_size: tuple[int, int],
    padding: tuple[int, int],
) -> None:
    if (
        not isinstance(projection, nn.Conv2d)
        or projection.in_channels != PATCH_INPUT_CHANNELS
        or projection.out_channels != PATCH_OUTPUT_CHANNELS
        or projection.kernel_size != kernel_size
        or projection.stride != PATCH_STRIDE
        or projection.padding != padding
        or projection.dilation != PATCH_DILATION
        or projection.groups != PATCH_GROUPS
        or (projection.bias is not None) is not PATCH_BIAS
        or projection.padding_mode != PATCH_PADDING_MODE
    ):
        raise RuntimeError("RGB patch-projection topology changed")


def _expand_patch_projection(predecessor: nn.Conv2d) -> nn.Conv2d:
    """Center-copy one exact 7x7 projection into an 11x11 projection."""

    _require_patch_projection(
        predecessor,
        kernel_size=PREDECESSOR_PATCH_KERNEL_SIZE,
        padding=(0, 0),
    )
    replacement = nn.Conv2d(
        PATCH_INPUT_CHANNELS,
        PATCH_OUTPUT_CHANNELS,
        kernel_size=PATCH_KERNEL_SIZE,
        stride=PATCH_STRIDE,
        padding=PATCH_PADDING,
        dilation=PATCH_DILATION,
        groups=PATCH_GROUPS,
        bias=PATCH_BIAS,
        padding_mode=PATCH_PADDING_MODE,
        device=predecessor.weight.device,
        dtype=predecessor.weight.dtype,
    )
    assert replacement.bias is not None and predecessor.bias is not None
    with torch.no_grad():
        replacement.weight.zero_()
        replacement.weight[:, :, 2:9, 2:9].copy_(predecessor.weight)
        replacement.bias.copy_(predecessor.bias)
    replacement.weight.requires_grad_(predecessor.weight.requires_grad)
    replacement.bias.requires_grad_(predecessor.bias.requires_grad)
    replacement.train(predecessor.training)
    return replacement


def _outer_ring_is_exact_zero(weight: torch.Tensor) -> bool:
    outer = weight.detach().clone()
    outer[:, :, 2:9, 2:9] = 0
    return int(torch.count_nonzero(outer).item()) == 0


@dataclass(frozen=True)
class OverlappingTokenizationInitializationReceiptV1:
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
    exact_copy_state_keys: tuple[str, ...]
    exact_copy_state_entry_count: int
    transformed_state_keys: tuple[str, ...]
    transformed_state_entry_count: int
    retained_n320_derived_entry_count: int
    source_patch_weight_shape: tuple[int, ...]
    destination_patch_weight_shape: tuple[int, ...]
    center_copy_slice: tuple[int, int, int, int]
    central_weight_scalar_count: int
    outer_ring_scalar_count: int
    patch_bias_scalar_count: int
    central_copy_exact: bool
    outer_ring_exact_zero: bool
    patch_bias_exact_copy: bool
    copied_predecessor_dense_decoder_entry_count: int
    canonical_ground_support_exact: bool
    hard_sync_count: int
    caller_cpu_rng_restored: bool
    replacement_module_caller_cpu_rng_restored: bool
    rejected_adaptation_checkpoint_open_count: int
    torch_version: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "copied_state_keys",
            "exact_copy_state_keys",
            "transformed_state_keys",
            "source_patch_weight_shape",
            "destination_patch_weight_shape",
            "center_copy_slice",
        ):
            value[key] = list(value[key])
        return value


class SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1(
    SharedObservableCameraRayJepaV5MultiresV1
):
    """Static multires Shared-V5 with overlapping first-layer support."""

    model_family = MODEL_FAMILY

    def __init__(
        self,
        config: SharedObservableCameraRayJepaV5Config | None = None,
    ) -> None:
        super().__init__(config=config)

        replacement_rng = torch.random.get_rng_state().clone()
        try:
            online_projection = _expand_patch_projection(
                self.encoder.patch_embed
            )
            target_projection = _expand_patch_projection(
                self.target_encoder.patch_embed
            )
        finally:
            torch.random.set_rng_state(replacement_rng)
        self._replacement_module_caller_cpu_rng_restored = torch.equal(
            torch.random.get_rng_state(),
            replacement_rng,
        )
        self.encoder.patch_embed = online_projection
        self.target_encoder.patch_embed = target_projection
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                name.startswith(("encoder.", "evidence_head."))
            )
        self.target_encoder.requires_grad_(False)
        self.target_bev_decoder.requires_grad_(False)
        self.target_encoder.eval()
        self.target_bev_decoder.eval()

        self._require_overlapping_tokenization_contract()
        selected = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        if self.model_config.encoder_depth == 6 and (
            _parameter_count(self.encoder) != EXPECTED_ENCODER_PARAMETER_COUNT
            or _parameter_tensor_count(self.encoder)
            != EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT
            or _parameter_count(self.evidence_head)
            != EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT
            or _parameter_tensor_count(self.evidence_head)
            != EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT
            or sum(int(parameter.numel()) for parameter in selected)
            != EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT
            or len(selected)
            != EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT
            or _parameter_count(self)
            != EXPECTED_COMPLETE_MODEL_PARAMETER_COUNT
            or _parameter_tensor_count(self)
            != EXPECTED_COMPLETE_MODEL_PARAMETER_TENSOR_COUNT
        ):
            raise RuntimeError(
                "overlapping-tokenization parameter contract changed"
            )

    def _require_overlapping_tokenization_contract(self) -> None:
        self._require_encoder_contract()
        if (
            self.encoder.patch_size != 7
            or self.target_encoder.patch_size != 7
            or self.encoder.num_patches != TOKEN_SIDE * TOKEN_SIDE
            or self.target_encoder.num_patches != TOKEN_SIDE * TOKEN_SIDE
        ):
            raise RuntimeError(
                "overlapping tokenization changed configured token geometry"
            )
        for projection in (
            self.encoder.patch_embed,
            self.target_encoder.patch_embed,
        ):
            _require_patch_projection(
                projection,
                kernel_size=PATCH_KERNEL_SIZE,
                padding=PATCH_PADDING,
            )
        if (
            self.encoder.patch_embed.weight.numel()
            != EXPECTED_PATCH_WEIGHT_PARAMETER_COUNT
            or self.encoder.pos_embed.shape
            != (1, TOKEN_SIDE * TOKEN_SIDE + 1, ENCODER_DIM)
            or self.target_encoder.training
            or any(
                parameter.requires_grad
                for parameter in self.target_encoder.parameters()
            )
        ):
            raise RuntimeError(
                "overlapping tokenization encoder contract changed"
            )

    def train(
        self, mode: bool = True
    ) -> "SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1":
        super().train(mode)
        self._require_overlapping_tokenization_contract()
        return self

    def migrate_from_fit_model(
        self,
        fit_model: ObservableCameraRayEvidenceV4Model,
    ) -> OverlappingTokenizationInitializationReceiptV1:
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
    ) -> OverlappingTokenizationInitializationReceiptV1:
        if not isinstance(fit_model, ObservableCameraRayEvidenceV4Model):
            raise TypeError(
                "fit_model must be an ObservableCameraRayEvidenceV4Model"
            )
        if (
            n320_checkpoint_file_sha256 != N320_CHECKPOINT_FILE_SHA256
            or n320_checkpoint_content_sha256
            != N320_CHECKPOINT_CONTENT_SHA256
        ):
            raise PermissionError("unbound fit checkpoint is not N320")
        if self._n320_initialization_complete:
            raise PermissionError("N320 initialization is one-shot")

        source_projection = fit_model.encoder.patch_embed
        destination_projection = self.encoder.patch_embed
        _require_patch_projection(
            source_projection,
            kernel_size=PREDECESSOR_PATCH_KERNEL_SIZE,
            padding=(0, 0),
        )
        _require_patch_projection(
            destination_projection,
            kernel_size=PATCH_KERNEL_SIZE,
            padding=PATCH_PADDING,
        )

        source_encoder_state = fit_model.encoder.state_dict()
        destination_encoder_state = self.encoder.state_dict()
        if set(source_encoder_state) != set(destination_encoder_state):
            raise RuntimeError("N320 encoder state-key inventory changed")
        transformed_local_key = "patch_embed.weight"
        migrated_encoder_state: dict[str, torch.Tensor] = {}
        exact_encoder_keys: list[str] = []
        for name, source in source_encoder_state.items():
            destination = destination_encoder_state[name]
            if name == transformed_local_key:
                if (
                    tuple(source.shape)
                    != (
                        PATCH_OUTPUT_CHANNELS,
                        PATCH_INPUT_CHANNELS,
                        *PREDECESSOR_PATCH_KERNEL_SIZE,
                    )
                    or tuple(destination.shape)
                    != (
                        PATCH_OUTPUT_CHANNELS,
                        PATCH_INPUT_CHANNELS,
                        *PATCH_KERNEL_SIZE,
                    )
                ):
                    raise RuntimeError(
                        "N320 patch-weight migration shape changed"
                    )
                expanded = torch.zeros_like(destination)
                expanded[:, :, 2:9, 2:9].copy_(source)
                migrated_encoder_state[name] = expanded
                continue
            if (
                source.shape != destination.shape
                or source.dtype != destination.dtype
            ):
                raise RuntimeError(
                    f"N320 exact-copy state contract changed for {name}"
                )
            migrated_encoder_state[name] = source
            exact_encoder_keys.append(f"encoder.{name}")

        self.encoder.load_state_dict(migrated_encoder_state, strict=True)
        copied_head = self.evidence_head.migrate_from_fit_model(fit_model)
        exact_copy_state_keys = tuple(
            sorted(
                (
                    *exact_encoder_keys,
                    *(
                        f"evidence_head.{name}"
                        for name in copied_head
                    ),
                )
            )
        )
        transformed_state_keys = ("encoder.patch_embed.weight",)
        copied_state_keys = tuple(
            sorted((*exact_copy_state_keys, *transformed_state_keys))
        )

        source_weight = source_encoder_state[transformed_local_key]
        destination_weight = self.encoder.patch_embed.weight.detach()
        assert (
            source_projection.bias is not None
            and self.encoder.patch_embed.bias is not None
        )
        central_copy_exact = torch.equal(
            destination_weight[:, :, 2:9, 2:9],
            source_weight,
        )
        outer_ring_exact_zero = _outer_ring_is_exact_zero(
            destination_weight
        )
        patch_bias_exact_copy = torch.equal(
            self.encoder.patch_embed.bias.detach(),
            source_projection.bias.detach(),
        )
        expected_exact_count = len(source_encoder_state) - 1 + 6
        expected_derived_count = len(source_encoder_state) + 6
        if (
            len(exact_copy_state_keys) != expected_exact_count
            or len(transformed_state_keys) != 1
            or len(copied_state_keys) != expected_derived_count
            or (
                self.model_config.encoder_depth == 6
                and (
                    len(exact_copy_state_keys) != 83
                    or len(copied_state_keys) != 84
                )
            )
            or not all(
                name.startswith(
                    (
                        "encoder.",
                        "evidence_head.pixel_head.",
                        "evidence_head.ground_head.",
                    )
                )
                for name in copied_state_keys
            )
            or any(
                "dense_decoder" in name for name in copied_state_keys
            )
            or not central_copy_exact
            or not outer_ring_exact_zero
            or not patch_bias_exact_copy
        ):
            raise RuntimeError("N320 overlap migration receipt changed")

        self.hard_sync_ema_target_from_online()
        self._require_overlapping_tokenization_contract()
        if tensor_state_dict_sha256(
            self.target_encoder.state_dict()
        ) != tensor_state_dict_sha256(self.encoder.state_dict()):
            raise RuntimeError("overlap target hard sync was not exact")
        self._n320_initialization_complete = True

        return OverlappingTokenizationInitializationReceiptV1(
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
            copied_state_keys=copied_state_keys,
            exact_copy_state_keys=exact_copy_state_keys,
            exact_copy_state_entry_count=len(exact_copy_state_keys),
            transformed_state_keys=transformed_state_keys,
            transformed_state_entry_count=1,
            retained_n320_derived_entry_count=len(copied_state_keys),
            source_patch_weight_shape=tuple(source_weight.shape),
            destination_patch_weight_shape=tuple(destination_weight.shape),
            center_copy_slice=CENTER_COPY_SLICE,
            central_weight_scalar_count=(
                EXPECTED_CENTRAL_WEIGHT_SCALAR_COUNT
            ),
            outer_ring_scalar_count=EXPECTED_OUTER_RING_SCALAR_COUNT,
            patch_bias_scalar_count=EXPECTED_PATCH_BIAS_SCALAR_COUNT,
            central_copy_exact=central_copy_exact,
            outer_ring_exact_zero=outer_ring_exact_zero,
            patch_bias_exact_copy=patch_bias_exact_copy,
            copied_predecessor_dense_decoder_entry_count=0,
            canonical_ground_support_exact=True,
            hard_sync_count=1,
            caller_cpu_rng_restored=True,
            replacement_module_caller_cpu_rng_restored=(
                self._replacement_module_caller_cpu_rng_restored
            ),
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
        "SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1",
        OverlappingTokenizationInitializationReceiptV1,
    ]:
        """Construct and one-shot migrate N320 under the frozen RNG boundary."""

        if not isinstance(fit_model, ObservableCameraRayEvidenceV4Model):
            raise TypeError(
                "fit_model must be an ObservableCameraRayEvidenceV4Model"
            )
        if (
            n320_checkpoint_file_sha256 != N320_CHECKPOINT_FILE_SHA256
            or n320_checkpoint_content_sha256
            != N320_CHECKPOINT_CONTENT_SHA256
        ):
            raise PermissionError("unbound fit checkpoint is not N320")

        caller_rng = torch.random.get_rng_state().clone()
        try:
            torch.manual_seed(BASE_INITIALIZATION_SEED)
            model = cls(config=config)
            receipt = model._migrate_from_n320_fit_model(
                fit_model,
                n320_checkpoint_file_sha256=n320_checkpoint_file_sha256,
                n320_checkpoint_content_sha256=(
                    n320_checkpoint_content_sha256
                ),
            )
        finally:
            torch.random.set_rng_state(caller_rng)
        if not torch.equal(torch.random.get_rng_state(), caller_rng):
            raise RuntimeError("N320 migration changed caller CPU RNG")
        return model, receipt


def overlapping_tokenization_architecture_contract_v1() -> dict[str, Any]:
    """Return the exact additive overlap architecture contract."""

    contract = multires_architecture_contract_v1()
    contract.update(
        {
            "schema": ARCHITECTURE_SCHEMA,
            "model_family": MODEL_FAMILY,
            "scientific_delta": ONE_SCIENCE_DELTA,
            "one_science_delta": ONE_SCIENCE_DELTA,
            "patch_projection": {
                "input_channels": PATCH_INPUT_CHANNELS,
                "output_channels": PATCH_OUTPUT_CHANNELS,
                "predecessor_kernel_size": list(
                    PREDECESSOR_PATCH_KERNEL_SIZE
                ),
                "kernel_size": list(PATCH_KERNEL_SIZE),
                "stride": list(PATCH_STRIDE),
                "padding": list(PATCH_PADDING),
                "dilation": list(PATCH_DILATION),
                "groups": PATCH_GROUPS,
                "bias": PATCH_BIAS,
                "padding_mode": PATCH_PADDING_MODE,
                "center_copy_slice": list(CENTER_COPY_SLICE),
                "central_weight_scalar_count": (
                    EXPECTED_CENTRAL_WEIGHT_SCALAR_COUNT
                ),
                "outer_ring_scalar_count": (
                    EXPECTED_OUTER_RING_SCALAR_COUNT
                ),
                "bias_scalar_count": EXPECTED_PATCH_BIAS_SCALAR_COUNT,
                "weight_parameter_count": (
                    EXPECTED_PATCH_WEIGHT_PARAMETER_COUNT
                ),
                "adjacent_overlap_pixels": 4,
                "configured_patch_size": 7,
            },
            "token_geometry": {
                "input_shape": [3, IMAGE_SIZE, IMAGE_SIZE],
                "patch_map_shape": [
                    PATCH_OUTPUT_CHANNELS,
                    TOKEN_SIDE,
                    TOKEN_SIDE,
                ],
                "patch_token_count": TOKEN_SIDE * TOKEN_SIDE,
                "patch_token_width": ENCODER_DIM,
                "cls_plus_patch_token_count": (
                    TOKEN_SIDE * TOKEN_SIDE + 1
                ),
                "positional_embedding_shape": [
                    1,
                    TOKEN_SIDE * TOKEN_SIDE + 1,
                    ENCODER_DIM,
                ],
                "token_center_formula": "7*i+3",
            },
            "trainable": {
                "encoder_parameter_count": EXPECTED_ENCODER_PARAMETER_COUNT,
                "encoder_parameter_tensor_count": (
                    EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT
                ),
                "evidence_head_parameter_count": (
                    EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT
                ),
                "evidence_head_parameter_tensor_count": (
                    EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT
                ),
                "total_parameter_count": (
                    EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT
                ),
                "total_parameter_tensor_count": (
                    EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT
                ),
            },
            "complete_model": {
                "parameter_count": EXPECTED_COMPLETE_MODEL_PARAMETER_COUNT,
                "parameter_tensor_count": (
                    EXPECTED_COMPLETE_MODEL_PARAMETER_TENSOR_COUNT
                ),
            },
            "jepa_tensor_interface": {
                "online_patch_tokens_shape": [256, 192],
                "target_patch_tokens_shape": [256, 192],
                "same_shape_ema_target": True,
            },
            "temporal_or_motion_module_present": False,
            "intermediate_encoder_features_used": False,
        }
    )
    return contract


__all__ = [
    "ARCHITECTURE_SCHEMA",
    "BASE_INITIALIZATION_SEED",
    "CENTER_COPY_SLICE",
    "DECODER_INITIALIZATION_SEED",
    "EXPECTED_CENTRAL_WEIGHT_SCALAR_COUNT",
    "EXPECTED_COMPLETE_MODEL_PARAMETER_COUNT",
    "EXPECTED_COMPLETE_MODEL_PARAMETER_TENSOR_COUNT",
    "EXPECTED_ENCODER_PARAMETER_COUNT",
    "EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT",
    "EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT",
    "EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT",
    "EXPECTED_OUTER_RING_SCALAR_COUNT",
    "EXPECTED_PATCH_BIAS_SCALAR_COUNT",
    "EXPECTED_PATCH_WEIGHT_PARAMETER_COUNT",
    "EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT",
    "EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT",
    "INITIALIZATION_SCHEMA",
    "MODEL_FAMILY",
    "N320_CHECKPOINT_CONTENT_SHA256",
    "N320_CHECKPOINT_FILE_SHA256",
    "ONE_SCIENCE_DELTA",
    "OverlappingTokenizationInitializationReceiptV1",
    "PATCH_BIAS",
    "PATCH_DILATION",
    "PATCH_GROUPS",
    "PATCH_KERNEL_SIZE",
    "PATCH_PADDING",
    "PATCH_PADDING_MODE",
    "PATCH_STRIDE",
    "SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1",
    "overlapping_tokenization_architecture_contract_v1",
]
