from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import OUTPUT_SHAPE
from lewm.models.observable_camera_ray_evidence_v4 import (
    IMAGE_SIZE,
    ObservableCameraRayEvidenceV4Model,
)
from lewm.models.shared_observable_camera_ray_jepa_v5 import (
    SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA,
    SharedObservableCameraRayJepaV5,
    SharedObservableCameraRayJepaV5Config,
    tensor_state_dict_sha256,
)
from lewm.models.shared_observable_camera_ray_jepa_v5_multires_v1 import (
    DECODER_INITIALIZATION_SEED,
    EXPECTED_DECODER_PARAMETER_COUNT,
    EXPECTED_DECODER_PARAMETER_TENSOR_COUNT,
    EXPECTED_ENCODER_PARAMETER_COUNT,
    EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT,
    EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT,
    EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT,
    EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT,
    EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT,
    MultiresDecoderV1Config,
    N320_CHECKPOINT_CONTENT_SHA256,
    N320_CHECKPOINT_FILE_SHA256,
    ProgressiveSpatialEvidenceDecoderV1,
    SharedObservableCameraRayJepaV5MultiresV1,
    multires_architecture_contract_v1,
)


def _parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def _parameter_tensor_count(module: nn.Module) -> int:
    return sum(1 for _parameter in module.parameters())


def _dev_config() -> SharedObservableCameraRayJepaV5Config:
    return SharedObservableCameraRayJepaV5Config(
        schema=SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA,
        encoder_depth=0,
        action_dim=3,
        bev_dim=8,
        bev_size=(4, 4),
        predictor_hidden_dim=12,
        target_ema_momentum=0.5,
        source_shape=(2, 3),
        pixel_ray_shape=(3, 4),
        query_chunk_size=5,
        v4_pixel_ray_chunk_size=32,
    )


def _calibration(batch: int = 1) -> tuple[torch.Tensor, ...]:
    origin = torch.tensor((0.326, 0.02, 0.043))[None].expand(batch, -1).clone()
    basis = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    )[None].expand(batch, -1, -1).clone()
    ground = torch.full((batch,), -0.35)
    return origin, basis, ground


def test_decoder_config_is_literal_and_immutable() -> None:
    config = MultiresDecoderV1Config()
    assert config.input_channels == 192
    assert config.stage_channels == (112, 80, 56, 36, 36)
    assert config.stage_sizes == (
        (16, 16),
        (28, 28),
        (56, 56),
        (112, 112),
        (112, 112),
    )
    assert config.group_counts == (8, 8, 8, 4, 4)
    assert config.initialization_seed == DECODER_INITIALIZATION_SEED
    with pytest.raises(FrozenInstanceError):
        config.padding = 0  # type: ignore[misc]
    with pytest.raises(ValueError, match="immutable"):
        replace(config, stage_channels=(112, 80, 56, 40, 36))


def test_decoder_topology_and_capacity_are_exact() -> None:
    decoder = ProgressiveSpatialEvidenceDecoderV1()
    convolutions = [
        module for module in decoder.modules() if isinstance(module, nn.Conv2d)
    ]
    normalizations = [
        module for module in decoder.modules() if isinstance(module, nn.GroupNorm)
    ]
    activations = [
        module for module in decoder.modules() if isinstance(module, nn.GELU)
    ]
    assert [
        (module.in_channels, module.out_channels)
        for module in convolutions
    ] == [(192, 112), (112, 80), (80, 56), (56, 36), (36, 36)]
    assert all(
        (
            module.kernel_size,
            module.stride,
            module.padding,
            module.padding_mode,
            module.bias is not None,
        )
        == ((3, 3), (1, 1), (1, 1), "zeros", True)
        for module in convolutions
    )
    assert [
        (module.num_groups, module.num_channels, module.eps, module.affine)
        for module in normalizations
    ] == [
        (8, 112, 1e-5, True),
        (8, 80, 1e-5, True),
        (8, 56, 1e-5, True),
        (4, 36, 1e-5, True),
        (4, 36, 1e-5, True),
    ]
    assert len(activations) == 5
    assert all(module.approximate == "none" for module in activations)
    assert _parameter_count(decoder) == EXPECTED_DECODER_PARAMETER_COUNT
    assert (
        _parameter_tensor_count(decoder)
        == EXPECTED_DECODER_PARAMETER_TENSOR_COUNT
    )


def test_decoder_initialization_is_local_deterministic_and_rng_neutral() -> None:
    torch.manual_seed(31)
    caller_rng = torch.random.get_rng_state().clone()
    first = ProgressiveSpatialEvidenceDecoderV1()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    torch.manual_seed(999)
    second_caller_rng = torch.random.get_rng_state().clone()
    second = ProgressiveSpatialEvidenceDecoderV1()
    assert torch.equal(torch.random.get_rng_state(), second_caller_rng)
    assert tensor_state_dict_sha256(first.state_dict()) == (
        tensor_state_dict_sha256(second.state_dict())
    )

    for module in first.modules():
        if isinstance(module, nn.Conv2d):
            assert module.bias is not None
            assert torch.count_nonzero(module.bias).item() == 0
        elif isinstance(module, nn.GroupNorm):
            assert module.weight is not None and module.bias is not None
            torch.testing.assert_close(
                module.weight,
                torch.ones_like(module.weight),
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                module.bias,
                torch.zeros_like(module.bias),
                rtol=0.0,
                atol=0.0,
            )


def test_decoder_forward_has_reviewed_shapes_and_gradients() -> None:
    decoder = ProgressiveSpatialEvidenceDecoderV1()
    token_map = torch.randn(1, 192, 16, 16, requires_grad=True)
    output, shapes = decoder.forward_with_shapes(token_map)
    assert output.shape == (1, 36, IMAGE_SIZE, IMAGE_SIZE)
    assert shapes == (
        (1, 112, 16, 16),
        (1, 80, 28, 28),
        (1, 56, 56, 56),
        (1, 36, 112, 112),
        (1, 36, 112, 112),
    )
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert token_map.grad is not None and torch.isfinite(token_map.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
    )


def test_decoder_uses_only_the_three_reviewed_explicit_resizes() -> None:
    decoder = ProgressiveSpatialEvidenceDecoderV1()
    original = torch.nn.functional.interpolate
    with mock.patch(
        "lewm.models.shared_observable_camera_ray_jepa_v5_multires_v1."
        "F.interpolate",
        wraps=original,
    ) as resize:
        decoder(torch.zeros(1, 192, 16, 16))
    assert [call.kwargs for call in resize.call_args_list] == [
        {
            "size": (28, 28),
            "mode": "bilinear",
            "align_corners": False,
            "antialias": False,
        },
        {
            "size": (56, 56),
            "mode": "bilinear",
            "align_corners": False,
            "antialias": False,
        },
        {
            "size": (112, 112),
            "mode": "bilinear",
            "align_corners": False,
            "antialias": False,
        },
    ]


@pytest.mark.parametrize(
    "value, match",
    [
        (torch.zeros(1, 191, 16, 16), "floating shape"),
        (torch.zeros(1, 192, 15, 16), "floating shape"),
        (torch.zeros(1, 192, 16, 16, dtype=torch.int64), "floating shape"),
    ],
)
def test_decoder_rejects_invalid_inputs(
    value: torch.Tensor, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ProgressiveSpatialEvidenceDecoderV1()(value)


def test_decoder_rejects_nonfinite_input() -> None:
    value = torch.zeros(1, 192, 16, 16)
    value[0, 0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        ProgressiveSpatialEvidenceDecoderV1()(value)


def test_successor_head_capacity_and_training_allowlist_capacity_are_exact() -> None:
    model = SharedObservableCameraRayJepaV5MultiresV1()
    assert _parameter_count(model.evidence_head) == (
        EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT
    )
    assert _parameter_tensor_count(model.evidence_head) == (
        EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT
    )
    assert _parameter_count(model.encoder) == EXPECTED_ENCODER_PARAMETER_COUNT
    assert _parameter_tensor_count(model.encoder) == (
        EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT
    )
    assert (
        _parameter_count(model.encoder) + _parameter_count(model.evidence_head)
        == EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT
    )
    assert (
        _parameter_tensor_count(model.encoder)
        + _parameter_tensor_count(model.evidence_head)
        == EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT
    )
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    frozen = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    assert all(
        name.startswith(("encoder.", "evidence_head."))
        for name, _parameter in trainable
    )
    assert all(
        name.startswith(
            (
                "bev_decoder.",
                "predictor.",
                "occupancy_head.",
                "target_encoder.",
                "target_bev_decoder.",
            )
        )
        for name, _parameter in frozen
    )
    assert sum(parameter.numel() for _name, parameter in trainable) == (
        EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT
    )
    assert len(trainable) == EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT


def test_migration_copies_only_encoder_and_compatible_consumers() -> None:
    torch.manual_seed(17)
    fit = ObservableCameraRayEvidenceV4Model(
        encoder_depth=0,
        source_shape=(2, 3),
        pixel_ray_shape=(3, 4),
        query_chunk_size=5,
    ).eval()
    torch.manual_seed(918)
    caller_rng = torch.random.get_rng_state().clone()
    successor, receipt = (
        SharedObservableCameraRayJepaV5MultiresV1.initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
            config=_dev_config(),
        )
    )
    successor.eval()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    assert tensor_state_dict_sha256(successor.encoder.state_dict()) == (
        tensor_state_dict_sha256(fit.encoder.state_dict())
    )
    assert tensor_state_dict_sha256(successor.evidence_head.pixel_head.state_dict()) == (
        tensor_state_dict_sha256(fit.pixel_head.state_dict())
    )
    assert tensor_state_dict_sha256(successor.evidence_head.ground_head.state_dict()) == (
        tensor_state_dict_sha256(fit.ground_head.state_dict())
    )
    assert tensor_state_dict_sha256(
        successor.evidence_head.dense_decoder.state_dict()
    ) == receipt.decoder_state_sha256
    assert receipt.copied_state_entry_count == (
        len(fit.encoder.state_dict()) + 6
    )
    assert receipt.copied_predecessor_dense_decoder_entry_count == 0
    assert all(
        name.startswith(
            (
                "encoder.",
                "evidence_head.pixel_head.",
                "evidence_head.ground_head.",
            )
        )
        for name in receipt.copied_state_keys
    )
    assert not any("dense_decoder" in name for name in receipt.copied_state_keys)
    assert receipt.initialization_input_role == "n320_fit_initialization_only"
    assert receipt.n320_checkpoint_file_sha256 == N320_CHECKPOINT_FILE_SHA256
    assert (
        receipt.n320_checkpoint_content_sha256
        == N320_CHECKPOINT_CONTENT_SHA256
    )
    assert receipt.canonical_ground_support_exact is True
    assert receipt.hard_sync_count == 1
    assert receipt.caller_cpu_rng_restored is True
    assert receipt.rejected_adaptation_checkpoint_open_count == 0
    assert tensor_state_dict_sha256(successor.target_encoder.state_dict()) == (
        tensor_state_dict_sha256(successor.encoder.state_dict())
    )


def test_migration_rejects_geometry_mismatch() -> None:
    fit = ObservableCameraRayEvidenceV4Model(
        encoder_depth=0,
        source_shape=(3, 3),
        pixel_ray_shape=(3, 4),
        query_chunk_size=5,
    )
    with pytest.raises(ValueError, match="geometries differ"):
        SharedObservableCameraRayJepaV5MultiresV1.initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
            config=_dev_config(),
        )


def test_generic_or_unbound_migration_is_prohibited_and_n320_is_one_shot() -> None:
    fit = ObservableCameraRayEvidenceV4Model(
        encoder_depth=0,
        source_shape=(2, 3),
        pixel_ray_shape=(3, 4),
        query_chunk_size=5,
    )
    model = SharedObservableCameraRayJepaV5MultiresV1(_dev_config())
    with pytest.raises(PermissionError, match="generic"):
        model.migrate_from_fit_model(fit)
    with pytest.raises(PermissionError, match="not N320"):
        model._migrate_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256="0" * 64,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
        )
    with pytest.raises(PermissionError, match="not N320"):
        SharedObservableCameraRayJepaV5MultiresV1.initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256="0" * 64,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
            config=_dev_config(),
        )
    initialized, _receipt = (
        SharedObservableCameraRayJepaV5MultiresV1.initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
            config=_dev_config(),
        )
    )
    with pytest.raises(PermissionError, match="one-shot"):
        initialized._migrate_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
        )


def test_n320_constructor_is_base_seeded_and_caller_rng_neutral() -> None:
    torch.manual_seed(73)
    fit = ObservableCameraRayEvidenceV4Model(
        encoder_depth=0,
        source_shape=(2, 3),
        pixel_ray_shape=(3, 4),
        query_chunk_size=5,
    )
    full_state_hashes = []
    for caller_seed in (1, 2):
        torch.manual_seed(caller_seed)
        caller_rng = torch.random.get_rng_state().clone()
        model, receipt = (
            SharedObservableCameraRayJepaV5MultiresV1.initialize_from_n320_fit_model(
                fit,
                n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
                n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
                config=_dev_config(),
            )
        )
        assert torch.equal(torch.random.get_rng_state(), caller_rng)
        assert receipt.base_initialization_seed == 20260712
        full_state_hashes.append(tensor_state_dict_sha256(model.state_dict()))
    assert full_state_hashes[0] == full_state_hashes[1]


def test_production_n320_copy_receipt_has_all_84_permitted_entries() -> None:
    fit = ObservableCameraRayEvidenceV4Model()
    _model, receipt = (
        SharedObservableCameraRayJepaV5MultiresV1.initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
        )
    )
    assert receipt.copied_state_entry_count == 84
    assert len(receipt.copied_state_keys) == 84


def test_predecessor_state_dict_cannot_strictly_load_into_successor() -> None:
    predecessor = SharedObservableCameraRayJepaV5(_dev_config())
    successor = SharedObservableCameraRayJepaV5MultiresV1(_dev_config())
    with pytest.raises(RuntimeError):
        successor.load_state_dict(predecessor.state_dict(), strict=True)


def test_successor_forward_preserves_evidence_output_contract() -> None:
    model = SharedObservableCameraRayJepaV5MultiresV1(_dev_config()).eval()
    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        evidence = model.forward_frame(image, *_calibration()).evidence
    assert evidence.pixel_first_hit_hazard_logits.shape == (1, 64, 3, 4)
    assert evidence.pixel_within_bin_offset_m.shape == (1, 64, 3, 4)
    assert evidence.ground_clear_to_target_logits.shape == (1, 2, 3, 5)
    assert evidence.ground_query_uv_px.shape == (1, 2, 3, 5, 2)
    assert evidence.ground_target_distance_m.shape == (1, 2, 3, 5)
    assert evidence.ground_query_in_frustum.shape == (1, 2, 3, 5)
    assert all(
        torch.isfinite(value).all()
        for value in (
            evidence.pixel_first_hit_hazard_logits,
            evidence.pixel_within_bin_offset_m,
            evidence.ground_clear_to_target_logits,
            evidence.ground_query_uv_px,
            evidence.ground_target_distance_m,
        )
    )


def test_forward_frame_encodes_online_once_and_never_calls_target_encoder() -> None:
    model = SharedObservableCameraRayJepaV5MultiresV1(_dev_config()).eval()
    with (
        mock.patch.object(
            model.encoder,
            "forward_tokens",
            wraps=model.encoder.forward_tokens,
        ) as online,
        mock.patch.object(
            model.target_encoder,
            "forward_tokens",
            wraps=model.target_encoder.forward_tokens,
        ) as target,
        torch.no_grad(),
    ):
        model.forward_frame(
            torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            *_calibration(),
        )
    assert online.call_count == 1
    assert target.call_count == 0


def test_successor_deployment_state_roundtrip_is_exact() -> None:
    model = SharedObservableCameraRayJepaV5MultiresV1(_dev_config())
    deployment = model.deployment_state_dict()
    assert all(
        name.startswith(("encoder.", "bev_decoder.", "evidence_head."))
        for name in deployment
    )
    restored = SharedObservableCameraRayJepaV5MultiresV1(_dev_config())
    restored.load_deployment_state_dict(deployment)
    assert tensor_state_dict_sha256(restored.deployment_state_dict()) == (
        tensor_state_dict_sha256(deployment)
    )


def test_synthetic_microfit_reduces_rgb_conditioned_evidence_error() -> None:
    torch.manual_seed(20260724)
    model = SharedObservableCameraRayJepaV5MultiresV1(_dev_config()).train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fitted = [
        *model.evidence_head.pixel_head.parameters(),
        *model.evidence_head.ground_head.parameters(),
    ]
    for parameter in fitted:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(fitted, lr=1e-2)
    image = torch.linspace(-1.0, 1.0, 3 * IMAGE_SIZE * IMAGE_SIZE).reshape(
        1, 3, IMAGE_SIZE, IMAGE_SIZE
    )

    def summary(value: torch.Tensor) -> torch.Tensor:
        evidence = model.forward_frame(value, *_calibration()).evidence
        return torch.stack(
            (
                evidence.pixel_first_hit_hazard_logits.mean(),
                evidence.ground_clear_to_target_logits.mean(),
            )
        )

    with torch.no_grad():
        initial = summary(image)
        target = initial + torch.tensor((0.35, -0.35))
        initial_loss = torch.nn.functional.mse_loss(initial, target)
    for _step in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(summary(image), target)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        matched = summary(image)
        wrong_rgb = summary(-image)
        final_loss = torch.nn.functional.mse_loss(matched, target)
    assert final_loss < initial_loss
    assert not torch.allclose(matched, wrong_rgb, rtol=0.0, atol=1e-6)


def test_architecture_contract_is_self_consistent() -> None:
    contract = multires_architecture_contract_v1()
    assert contract["decoder"]["stage_channels"] == [112, 80, 56, 36, 36]
    assert contract["decoder"]["stage_sizes"] == [
        [16, 16],
        [28, 28],
        [56, 56],
        [112, 112],
        [112, 112],
    ]
    assert contract["decoder"]["resize_mode"] == "bilinear"
    assert contract["decoder"]["align_corners"] is False
    assert contract["decoder"]["antialias"] is False
    assert contract["decoder"]["parameter_count"] == (
        EXPECTED_DECODER_PARAMETER_COUNT
    )
    assert contract["trainable"]["total_parameter_count"] == (
        EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT
    )
    assert contract["unchanged_consumers"] == [
        "pixel_head",
        "ground_head",
        "camera_geometry",
        "ray_depth_ground_output_contract",
        "rasterization",
    ]
    assert contract["intermediate_encoder_features_used"] is False
    assert OUTPUT_SHAPE == (64, 64)
