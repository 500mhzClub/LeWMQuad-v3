from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""
os.environ["GPU_DEVICE_ORDINAL"] = ""

import pytest
import torch
import torch.nn as nn

from lewm.models.observable_camera_ray_evidence_v4 import (
    IMAGE_SIZE,
    ObservableCameraRayEvidenceV4Model,
)
from lewm.models.shared_observable_camera_ray_jepa_v5 import (
    SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA,
    SharedObservableCameraRayJepaV5Config,
    tensor_state_dict_sha256,
)
from lewm.models.shared_observable_camera_ray_jepa_v5_multires_overlapping_tokenization_v1 import (
    ARCHITECTURE_SCHEMA,
    CENTER_COPY_SLICE,
    EXPECTED_CENTRAL_WEIGHT_SCALAR_COUNT,
    EXPECTED_COMPLETE_MODEL_PARAMETER_COUNT,
    EXPECTED_COMPLETE_MODEL_PARAMETER_TENSOR_COUNT,
    EXPECTED_ENCODER_PARAMETER_COUNT,
    EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT,
    EXPECTED_OUTER_RING_SCALAR_COUNT,
    EXPECTED_PATCH_BIAS_SCALAR_COUNT,
    EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT,
    EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT,
    MODEL_FAMILY,
    N320_CHECKPOINT_CONTENT_SHA256,
    N320_CHECKPOINT_FILE_SHA256,
    ONE_SCIENCE_DELTA,
    SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1,
    overlapping_tokenization_architecture_contract_v1,
)
from lewm.models.shared_observable_camera_ray_jepa_v5_multires_v1 import (
    EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT,
    EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT,
    SharedObservableCameraRayJepaV5MultiresV1,
)


def _parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def _parameter_tensor_count(module: nn.Module) -> int:
    return sum(1 for _parameter in module.parameters())


def _dev_config(
    *,
    target_ema_momentum: float = 0.5,
) -> SharedObservableCameraRayJepaV5Config:
    return SharedObservableCameraRayJepaV5Config(
        schema=SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA,
        encoder_depth=0,
        action_dim=3,
        bev_dim=8,
        bev_size=(4, 4),
        predictor_hidden_dim=12,
        target_ema_momentum=target_ema_momentum,
        source_shape=(2, 3),
        pixel_ray_shape=(3, 4),
        query_chunk_size=5,
        v4_pixel_ray_chunk_size=32,
    )


def _fit_model(
    config: SharedObservableCameraRayJepaV5Config,
) -> ObservableCameraRayEvidenceV4Model:
    return ObservableCameraRayEvidenceV4Model(
        encoder_depth=config.encoder_depth,
        source_shape=config.source_shape,
        pixel_ray_shape=config.pixel_ray_shape,
        query_chunk_size=config.query_chunk_size,
    )


def _outer_ring(weight: torch.Tensor) -> torch.Tensor:
    output = weight.detach().clone()
    output[:, :, 2:9, 2:9] = 0
    return output


def test_patch_topology_center_copy_rng_and_update_zero_are_exact() -> None:
    config = _dev_config()
    torch.manual_seed(91)
    predecessor = SharedObservableCameraRayJepaV5MultiresV1(config)
    predecessor_post_constructor_rng = torch.random.get_rng_state().clone()

    torch.manual_seed(91)
    successor = (
        SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1(
            config
        )
    )
    assert torch.equal(
        torch.random.get_rng_state(),
        predecessor_post_constructor_rng,
    )

    old = predecessor.encoder.patch_embed
    new = successor.encoder.patch_embed
    assert (
        new.in_channels,
        new.out_channels,
        new.kernel_size,
        new.stride,
        new.padding,
        new.dilation,
        new.groups,
        new.bias is not None,
        new.padding_mode,
    ) == (
        3,
        192,
        (11, 11),
        (7, 7),
        (2, 2),
        (1, 1),
        1,
        True,
        "zeros",
    )
    assert successor.encoder.patch_size == 7
    assert successor.encoder.num_patches == 256
    assert successor.encoder.pos_embed.shape == (1, 257, 192)
    assert torch.equal(new.weight[:, :, 2:9, 2:9], old.weight)
    assert torch.count_nonzero(_outer_ring(new.weight)).item() == 0
    assert new.bias is not None and old.bias is not None
    assert torch.equal(new.bias, old.bias)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(827)
    old_input = torch.randn(
        2,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        generator=generator,
        requires_grad=True,
    )
    new_input = old_input.detach().clone().requires_grad_(True)
    old_output = old(old_input)
    new_output = new(new_input)
    assert old_output.shape == new_output.shape == (2, 192, 16, 16)
    torch.testing.assert_close(
        new_output,
        old_output,
        rtol=1e-6,
        atol=1e-6,
    )

    upstream = torch.randn(
        old_output.shape,
        generator=generator,
        dtype=old_output.dtype,
    )
    old_output.backward(upstream)
    new_output.backward(upstream)
    assert old_input.grad is not None and new_input.grad is not None
    torch.testing.assert_close(
        new_input.grad,
        old_input.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    assert old.weight.grad is not None and new.weight.grad is not None
    torch.testing.assert_close(
        new.weight.grad[:, :, 2:9, 2:9],
        old.weight.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    outer_gradient = _outer_ring(new.weight.grad)
    assert torch.isfinite(outer_gradient).all()
    assert torch.count_nonzero(outer_gradient).item() > 0


def test_architecture_contract_binds_topology_geometry_and_capacity() -> None:
    contract = overlapping_tokenization_architecture_contract_v1()
    assert contract["schema"] == ARCHITECTURE_SCHEMA
    assert contract["model_family"] == MODEL_FAMILY
    assert contract["scientific_delta"] == ONE_SCIENCE_DELTA
    assert contract["one_science_delta"] == ONE_SCIENCE_DELTA
    assert contract["patch_projection"] == {
        "input_channels": 3,
        "output_channels": 192,
        "predecessor_kernel_size": [7, 7],
        "kernel_size": [11, 11],
        "stride": [7, 7],
        "padding": [2, 2],
        "dilation": [1, 1],
        "groups": 1,
        "bias": True,
        "padding_mode": "zeros",
        "center_copy_slice": [2, 9, 2, 9],
        "central_weight_scalar_count": (
            EXPECTED_CENTRAL_WEIGHT_SCALAR_COUNT
        ),
        "outer_ring_scalar_count": EXPECTED_OUTER_RING_SCALAR_COUNT,
        "bias_scalar_count": EXPECTED_PATCH_BIAS_SCALAR_COUNT,
        "weight_parameter_count": 69_696,
        "adjacent_overlap_pixels": 4,
        "configured_patch_size": 7,
    }
    assert contract["token_geometry"] == {
        "input_shape": [3, 112, 112],
        "patch_map_shape": [192, 16, 16],
        "patch_token_count": 256,
        "patch_token_width": 192,
        "cls_plus_patch_token_count": 257,
        "positional_embedding_shape": [1, 257, 192],
        "token_center_formula": "7*i+3",
    }
    assert contract["trainable"] == {
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
    }
    assert contract["complete_model"] == {
        "parameter_count": EXPECTED_COMPLETE_MODEL_PARAMETER_COUNT,
        "parameter_tensor_count": (
            EXPECTED_COMPLETE_MODEL_PARAMETER_TENSOR_COUNT
        ),
    }
    assert contract["temporal_or_motion_module_present"] is False


def test_production_n320_migration_receipt_is_exact_83_plus_one() -> None:
    torch.manual_seed(701)
    fit = ObservableCameraRayEvidenceV4Model()
    source_weight = fit.encoder.patch_embed.weight.detach().clone()
    assert fit.encoder.patch_embed.bias is not None
    source_bias = fit.encoder.patch_embed.bias.detach().clone()

    torch.manual_seed(119)
    caller_rng = torch.random.get_rng_state().clone()
    model, receipt = (
        SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1
        .initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=(
                N320_CHECKPOINT_CONTENT_SHA256
            ),
        )
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    assert receipt.model_family == MODEL_FAMILY
    assert receipt.exact_copy_state_entry_count == 83
    assert len(receipt.exact_copy_state_keys) == 83
    assert "encoder.patch_embed.bias" in receipt.exact_copy_state_keys
    assert receipt.transformed_state_keys == (
        "encoder.patch_embed.weight",
    )
    assert receipt.transformed_state_entry_count == 1
    assert receipt.retained_n320_derived_entry_count == 84
    assert len(receipt.copied_state_keys) == 84
    assert receipt.source_patch_weight_shape == (192, 3, 7, 7)
    assert receipt.destination_patch_weight_shape == (192, 3, 11, 11)
    assert receipt.center_copy_slice == CENTER_COPY_SLICE
    assert (
        receipt.central_weight_scalar_count
        == EXPECTED_CENTRAL_WEIGHT_SCALAR_COUNT
    )
    assert (
        receipt.outer_ring_scalar_count
        == EXPECTED_OUTER_RING_SCALAR_COUNT
    )
    assert (
        receipt.patch_bias_scalar_count
        == EXPECTED_PATCH_BIAS_SCALAR_COUNT
    )
    assert receipt.central_copy_exact is True
    assert receipt.outer_ring_exact_zero is True
    assert receipt.patch_bias_exact_copy is True
    assert receipt.copied_predecessor_dense_decoder_entry_count == 0
    assert receipt.hard_sync_count == 1
    assert receipt.caller_cpu_rng_restored is True
    assert receipt.replacement_module_caller_cpu_rng_restored is True
    assert receipt.rejected_adaptation_checkpoint_open_count == 0

    assert torch.equal(
        model.encoder.patch_embed.weight[:, :, 2:9, 2:9],
        source_weight,
    )
    assert torch.count_nonzero(
        _outer_ring(model.encoder.patch_embed.weight)
    ).item() == 0
    assert model.encoder.patch_embed.bias is not None
    assert torch.equal(model.encoder.patch_embed.bias, source_bias)
    assert tensor_state_dict_sha256(
        model.target_encoder.state_dict()
    ) == tensor_state_dict_sha256(model.encoder.state_dict())
    assert model.target_encoder.training is False
    assert not any(
        parameter.requires_grad
        for parameter in model.target_encoder.parameters()
    )

    assert _parameter_count(model.encoder) == (
        EXPECTED_ENCODER_PARAMETER_COUNT
    )
    assert _parameter_tensor_count(model.encoder) == (
        EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT
    )
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert sum(
        int(parameter.numel()) for _name, parameter in trainable
    ) == EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT
    assert len(trainable) == EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT
    assert _parameter_count(model) == EXPECTED_COMPLETE_MODEL_PARAMETER_COUNT
    assert (
        _parameter_tensor_count(model)
        == EXPECTED_COMPLETE_MODEL_PARAMETER_TENSOR_COUNT
    )


def test_jepa_shapes_one_ema_step_and_same_family_roundtrip() -> None:
    config = _dev_config(target_ema_momentum=0.5)
    torch.manual_seed(307)
    fit = _fit_model(config)
    model, _receipt = (
        SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1
        .initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=(
                N320_CHECKPOINT_CONTENT_SHA256
            ),
            config=config,
        )
    )
    model.eval()
    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        online = model.encoder.forward_tokens(image)
        target = model.target_encoder.forward_tokens(image)
    assert online.shape == target.shape == (1, 257, 192)
    torch.testing.assert_close(online, target, rtol=0.0, atol=0.0)

    with torch.no_grad():
        model.encoder.patch_embed.weight[0, 0, 0, 0] = 1.0
    assert model.target_encoder.patch_embed.weight[0, 0, 0, 0].item() == 0.0
    model.update_ema_target_after_optimizer_step()
    assert model.target_encoder.patch_embed.weight[
        0, 0, 0, 0
    ].item() == pytest.approx(0.5, rel=0.0, abs=0.0)
    assert model.target_encoder.training is False
    assert not any(
        parameter.requires_grad
        for parameter in model.target_encoder.parameters()
    )

    deployment = model.deployment_state_dict()
    restored = (
        SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1(
            config
        )
    )
    restored.load_deployment_state_dict(deployment)
    assert tensor_state_dict_sha256(
        restored.deployment_state_dict()
    ) == tensor_state_dict_sha256(deployment)
    assert restored.encoder.patch_embed.weight.shape == (192, 3, 11, 11)

    old_family = SharedObservableCameraRayJepaV5MultiresV1(config)
    with pytest.raises(RuntimeError):
        old_family.load_state_dict(model.state_dict(), strict=True)


def test_unbound_generic_and_repeated_n320_migration_are_rejected() -> None:
    config = _dev_config()
    fit = _fit_model(config)
    model = (
        SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1(
            config
        )
    )
    with pytest.raises(PermissionError, match="generic"):
        model.migrate_from_fit_model(fit)
    with pytest.raises(PermissionError, match="not N320"):
        model._migrate_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256="0" * 64,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
        )
    initialized, _receipt = (
        SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1
        .initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=(
                N320_CHECKPOINT_CONTENT_SHA256
            ),
            config=config,
        )
    )
    with pytest.raises(PermissionError, match="one-shot"):
        initialized._migrate_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=(
                N320_CHECKPOINT_CONTENT_SHA256
            ),
        )
