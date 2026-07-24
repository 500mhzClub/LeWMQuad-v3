from __future__ import annotations

from unittest import mock

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
from lewm.models.shared_observable_camera_ray_jepa_v5_multires_temporal_v1 import (
    EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT,
    EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT,
    EXPECTED_TEMPORAL_PARAMETER_COUNT,
    EXPECTED_TEMPORAL_PARAMETER_TENSOR_COUNT,
    EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT,
    EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT,
    HISTORY_LAG_SECONDS,
    HISTORY_LAG_TICKS,
    N320_CHECKPOINT_CONTENT_SHA256,
    N320_CHECKPOINT_FILE_SHA256,
    SharedObservableCameraRayJepaV5MultiresTemporalV1,
    TEMPORAL_INITIALIZATION_SEED,
    VisualTokenDifferenceResidualV1,
    temporal_multires_architecture_contract_v1,
)
from lewm.models.shared_observable_camera_ray_jepa_v5_multires_v1 import (
    EXPECTED_ENCODER_PARAMETER_COUNT,
    EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT,
    PREDECESSOR_EVIDENCE_HEAD_PARAMETER_CEILING,
    SharedObservableCameraRayJepaV5MultiresV1,
)


def _parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def _parameter_tensor_count(module: nn.Module) -> int:
    return sum(1 for _parameter in module.parameters())


def _dev_config(
    *,
    encoder_depth: int = 0,
) -> SharedObservableCameraRayJepaV5Config:
    return SharedObservableCameraRayJepaV5Config(
        schema=SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA,
        encoder_depth=encoder_depth,
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


def _images(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(811)
    previous = torch.randn(
        batch,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        generator=generator,
    )
    current = torch.randn(
        batch,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        generator=generator,
    )
    return previous, current


def _evidence_tensors(value: object) -> tuple[torch.Tensor, ...]:
    return (
        value.pixel_first_hit_hazard_logits,
        value.pixel_within_bin_offset_m,
        value.ground_clear_to_target_logits,
        value.ground_query_in_frustum,
        value.ground_query_uv_px,
        value.ground_target_distance_m,
    )


def _assert_evidence_equal(left: object, right: object) -> None:
    for observed, expected in zip(
        _evidence_tensors(left),
        _evidence_tensors(right),
        strict=True,
    ):
        assert torch.equal(observed, expected)


def _fit_model(
    config: SharedObservableCameraRayJepaV5Config,
) -> ObservableCameraRayEvidenceV4Model:
    return ObservableCameraRayEvidenceV4Model(
        encoder_depth=config.encoder_depth,
        source_shape=config.source_shape,
        pixel_ray_shape=config.pixel_ray_shape,
        query_chunk_size=config.query_chunk_size,
    )


def test_temporal_topology_capacity_initialization_and_rng_are_exact() -> None:
    torch.manual_seed(31)
    caller_rng = torch.random.get_rng_state().clone()
    first = VisualTokenDifferenceResidualV1()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    torch.manual_seed(999)
    second_caller_rng = torch.random.get_rng_state().clone()
    second = VisualTokenDifferenceResidualV1()
    assert torch.equal(torch.random.get_rng_state(), second_caller_rng)
    assert tensor_state_dict_sha256(first.state_dict()) == (
        tensor_state_dict_sha256(second.state_dict())
    )

    assert tuple(first.state_dict()) == (
        "input_projection.weight",
        "normalization.weight",
        "normalization.bias",
        "spatial_projection.weight",
        "output_projection.weight",
    )
    assert (
        first.input_projection.in_channels,
        first.input_projection.out_channels,
        first.input_projection.kernel_size,
        first.input_projection.stride,
        first.input_projection.padding,
        first.input_projection.bias,
    ) == (192, 8, (1, 1), (1, 1), (0, 0), None)
    assert (
        first.normalization.num_groups,
        first.normalization.num_channels,
        first.normalization.eps,
        first.normalization.affine,
    ) == (4, 8, 1e-5, True)
    assert first.activation.approximate == "none"
    assert (
        first.spatial_projection.in_channels,
        first.spatial_projection.out_channels,
        first.spatial_projection.kernel_size,
        first.spatial_projection.stride,
        first.spatial_projection.padding,
        first.spatial_projection.groups,
        first.spatial_projection.padding_mode,
        first.spatial_projection.bias,
    ) == (8, 8, (3, 3), (1, 1), (1, 1), 8, "zeros", None)
    assert (
        first.output_projection.in_channels,
        first.output_projection.out_channels,
        first.output_projection.kernel_size,
        first.output_projection.bias,
    ) == (8, 192, (1, 1), None)
    assert torch.equal(
        first.normalization.weight,
        torch.ones_like(first.normalization.weight),
    )
    assert torch.equal(
        first.normalization.bias,
        torch.zeros_like(first.normalization.bias),
    )
    assert torch.count_nonzero(first.input_projection.weight).item() > 0
    assert torch.count_nonzero(first.spatial_projection.weight).item() > 0
    assert torch.count_nonzero(first.output_projection.weight).item() == 0
    assert _parameter_count(first) == EXPECTED_TEMPORAL_PARAMETER_COUNT
    assert (
        _parameter_tensor_count(first)
        == EXPECTED_TEMPORAL_PARAMETER_TENSOR_COUNT
    )


def test_temporal_tokens_validate_and_mixed_cold_rows_bypass_exactly() -> None:
    block = VisualTokenDifferenceResidualV1()
    with torch.no_grad():
        block.output_projection.weight.fill_(0.01)
    previous = torch.randn(2, 256, 192)
    current = torch.randn(2, 256, 192)
    fused = block(
        previous,
        current,
        torch.tensor((False, True)),
    )
    assert torch.equal(fused[0], current[0])
    assert not torch.equal(fused[1], current[1])
    assert torch.isfinite(fused).all()

    with pytest.raises(ValueError, match="matching floating shape"):
        block(
            torch.zeros(1, 255, 192),
            torch.zeros(1, 255, 192),
            torch.zeros(1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="history_valid"):
        block(
            torch.zeros(1, 256, 192),
            torch.zeros(1, 256, 192),
            torch.zeros(1),
        )
    nonfinite = torch.zeros(1, 256, 192)
    nonfinite[0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        block(
            nonfinite,
            torch.zeros(1, 256, 192),
            torch.ones(1, dtype=torch.bool),
        )


def test_production_capacity_partition_and_architecture_contract_are_exact() -> None:
    model = SharedObservableCameraRayJepaV5MultiresTemporalV1()
    assert _parameter_count(model.temporal_residual) == (
        EXPECTED_TEMPORAL_PARAMETER_COUNT
    )
    assert _parameter_tensor_count(model.temporal_residual) == (
        EXPECTED_TEMPORAL_PARAMETER_TENSOR_COUNT
    )
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
    assert sum(parameter.numel() for _name, parameter in trainable) == (
        EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT
    )
    assert len(trainable) == EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT
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
    assert (
        PREDECESSOR_EVIDENCE_HEAD_PARAMETER_CEILING
        - EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT
        == 2_144
    )

    contract = temporal_multires_architecture_contract_v1()
    assert contract["history"] == {
        "lag_seconds": HISTORY_LAG_SECONDS,
        "lag_ticks": HISTORY_LAG_TICKS,
        "caller_supplied_history_valid": True,
        "model_owned_history_buffer": False,
        "raw_tokens_only": True,
    }
    assert contract["temporal"]["initialization_seed"] == (
        TEMPORAL_INITIALIZATION_SEED
    )
    assert contract["temporal"]["parameter_count"] == 3_160
    assert contract["temporal"]["parameter_tensor_count"] == 5
    assert contract["temporal"]["context_inputs"] == []
    assert contract["separate_online_encoder_calls"] is True
    assert contract["target_encoder_calls"] == 0
    assert contract["jepa_objective_count"] == 0


def test_update_zero_matches_independently_initialized_multires_predecessor() -> None:
    config = _dev_config()
    torch.manual_seed(17)
    fit = _fit_model(config).eval()
    predecessor, _predecessor_receipt = (
        SharedObservableCameraRayJepaV5MultiresV1.initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
            config=config,
        )
    )
    temporal, receipt = (
        SharedObservableCameraRayJepaV5MultiresTemporalV1
        .initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
            config=config,
        )
    )
    predecessor.eval()
    temporal.eval()

    predecessor_state = predecessor.state_dict()
    temporal_state = temporal.state_dict()
    assert all(
        name in temporal_state and torch.equal(value, temporal_state[name])
        for name, value in predecessor_state.items()
    )
    assert set(temporal_state) - set(predecessor_state) == {
        "evidence_head.temporal_residual.input_projection.weight",
        "evidence_head.temporal_residual.normalization.weight",
        "evidence_head.temporal_residual.normalization.bias",
        "evidence_head.temporal_residual.spatial_projection.weight",
        "evidence_head.temporal_residual.output_projection.weight",
    }

    previous_image, current_image = _images()
    calibration = _calibration()
    with torch.no_grad():
        expected_previous = predecessor.forward_frame(
            previous_image,
            *calibration,
        )
        expected_current = predecessor.forward_frame(
            current_image,
            *calibration,
        )
        observed_previous, observed_current = temporal.forward_camera_pair(
            previous_image,
            current_image,
            *calibration,
            *calibration,
        )
    assert torch.equal(
        observed_previous.patch_tokens,
        expected_previous.patch_tokens,
    )
    assert torch.equal(observed_previous.bev, expected_previous.bev)
    _assert_evidence_equal(
        observed_previous.evidence,
        expected_previous.evidence,
    )
    assert torch.equal(observed_current.patch_tokens, expected_current.patch_tokens)
    assert torch.equal(observed_current.bev, expected_current.bev)
    _assert_evidence_equal(observed_current.evidence, expected_current.evidence)
    assert receipt.temporal_initialization_seed == TEMPORAL_INITIALIZATION_SEED
    assert receipt.temporal_state_sha256 == tensor_state_dict_sha256(
        temporal.temporal_residual.state_dict()
    )
    assert receipt.copied_temporal_entry_count == 0
    assert receipt.temporal_output_projection_exact_zero is True


def test_pair_apis_use_two_separate_online_encodes_and_no_target_encode() -> None:
    model = SharedObservableCameraRayJepaV5MultiresTemporalV1(
        _dev_config()
    ).eval()
    previous_image, current_image = _images()
    calibration = _calibration()
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
        previous_frame, current_frame = model.forward_camera_pair(
            previous_image,
            current_image,
            *calibration,
            *calibration,
        )
    assert online.call_count == 2
    assert torch.equal(online.call_args_list[0].args[0], previous_image)
    assert torch.equal(online.call_args_list[1].args[0], current_image)
    assert target.call_count == 0
    assert previous_frame.patch_tokens.shape == (1, 256, 192)
    assert current_frame.patch_tokens.shape == (1, 256, 192)

    with pytest.raises(PermissionError, match="bypasses temporal fusion"):
        model.forward_training_pair()


def test_learned_temporal_state_keeps_forward_frame_and_cold_rows_exact() -> None:
    model = SharedObservableCameraRayJepaV5MultiresTemporalV1(
        _dev_config()
    ).eval()
    with torch.no_grad():
        model.temporal_residual.output_projection.weight.fill_(0.01)
    previous_a, current = _images()
    previous_b = -previous_a
    calibration = _calibration()
    cold = torch.zeros(1, dtype=torch.bool)
    warm = torch.ones(1, dtype=torch.bool)
    with torch.no_grad():
        baseline = model.forward_frame(current, *calibration)
        cold_a = model.forward_temporal_frame(
            previous_a,
            current,
            *calibration,
            cold,
        )
        cold_b = model.forward_temporal_frame(
            previous_b,
            current,
            *calibration,
            cold,
        )
        warm_a = model.forward_temporal_frame(
            previous_a,
            current,
            *calibration,
            warm,
        )
    assert torch.equal(cold_a.patch_tokens, baseline.patch_tokens)
    assert torch.equal(cold_a.bev, baseline.bev)
    _assert_evidence_equal(cold_a.evidence, baseline.evidence)
    _assert_evidence_equal(cold_b.evidence, baseline.evidence)
    assert any(
        not torch.equal(observed, expected)
        for observed, expected in zip(
            _evidence_tensors(warm_a.evidence)[:3],
            _evidence_tensors(baseline.evidence)[:3],
            strict=True,
        )
    )


def test_zero_projection_gradient_then_all_temporal_gradients_after_one_step() -> None:
    block = VisualTokenDifferenceResidualV1()
    optimizer = torch.optim.SGD(block.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260725)

    previous = torch.randn(
        2, 256, 192, generator=generator, requires_grad=True
    )
    current = torch.randn(
        2, 256, 192, generator=generator, requires_grad=True
    )
    probe = torch.randn(2, 256, 192, generator=generator)
    first = (block(
        previous,
        current,
        torch.ones(2, dtype=torch.bool),
    ) * probe).mean()
    first.backward()

    upstream = (
        block.input_projection.weight,
        block.normalization.weight,
        block.normalization.bias,
        block.spatial_projection.weight,
    )
    assert block.output_projection.weight.grad is not None
    assert torch.isfinite(block.output_projection.weight.grad).all()
    assert torch.count_nonzero(block.output_projection.weight.grad).item() > 0
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() == 0
        for parameter in upstream
    )
    assert previous.grad is not None
    assert torch.count_nonzero(previous.grad).item() == 0
    optimizer.step()
    assert torch.count_nonzero(block.output_projection.weight).item() > 0

    optimizer.zero_grad(set_to_none=True)
    previous_second = torch.randn(
        2, 256, 192, generator=generator, requires_grad=True
    )
    current_second = torch.randn(
        2, 256, 192, generator=generator, requires_grad=True
    )
    probe_second = torch.randn(2, 256, 192, generator=generator)
    second = (block(
        previous_second,
        current_second,
        torch.ones(2, dtype=torch.bool),
    ) * probe_second).mean()
    second.backward()
    assert previous_second.grad is not None
    assert torch.isfinite(previous_second.grad).all()
    assert torch.count_nonzero(previous_second.grad).item() > 0
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in block.parameters()
    )


def test_full_pair_backward_reaches_every_trainable_tensor_and_no_frozen_tensor() -> None:
    model = SharedObservableCameraRayJepaV5MultiresTemporalV1(
        _dev_config(encoder_depth=1)
    ).train()
    previous_image, current_image = _images()
    previous_frame, current_frame = model.forward_camera_pair(
        previous_image,
        current_image,
        *_calibration(),
        *_calibration(),
    )
    loss = sum(
        (
            frame.evidence.pixel_first_hit_hazard_logits.square().mean()
            + frame.evidence.ground_clear_to_target_logits.square().mean()
        )
        for frame in (previous_frame, current_frame)
    )
    loss.backward()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    frozen = [
        parameter for parameter in model.parameters() if not parameter.requires_grad
    ]
    assert trainable
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainable
    )
    assert all(parameter.grad is None for parameter in frozen)


def test_n320_receipt_copies_84_entries_and_never_copies_temporal_state() -> None:
    torch.manual_seed(73)
    fit = ObservableCameraRayEvidenceV4Model()
    torch.manual_seed(918)
    caller_rng = torch.random.get_rng_state().clone()
    model, receipt = (
        SharedObservableCameraRayJepaV5MultiresTemporalV1
        .initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
        )
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert receipt.copied_state_entry_count == 84
    assert len(receipt.copied_state_keys) == 84
    assert receipt.copied_predecessor_dense_decoder_entry_count == 0
    assert receipt.copied_temporal_entry_count == 0
    assert not any(
        "dense_decoder" in name or "temporal_residual" in name
        for name in receipt.copied_state_keys
    )
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
    assert receipt.hard_sync_count == 1
    assert receipt.caller_cpu_rng_restored is True
    assert receipt.rejected_adaptation_checkpoint_open_count == 0
    assert receipt.temporal_output_projection_exact_zero is True
    assert tensor_state_dict_sha256(model.encoder.state_dict()) == (
        tensor_state_dict_sha256(model.target_encoder.state_dict())
    )
    assert receipt.temporal_state_sha256 == tensor_state_dict_sha256(
        model.temporal_residual.state_dict()
    )


def test_generic_unbound_and_repeated_n320_migration_are_prohibited() -> None:
    config = _dev_config()
    fit = _fit_model(config)
    model = SharedObservableCameraRayJepaV5MultiresTemporalV1(config)
    with pytest.raises(PermissionError, match="generic"):
        model.migrate_from_fit_model(fit)
    with pytest.raises(PermissionError, match="not N320"):
        (
            SharedObservableCameraRayJepaV5MultiresTemporalV1
            .initialize_from_n320_fit_model(
                fit,
                n320_checkpoint_file_sha256="0" * 64,
                n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
                config=config,
            )
        )
    initialized, _receipt = (
        SharedObservableCameraRayJepaV5MultiresTemporalV1
        .initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
            config=config,
        )
    )
    with pytest.raises(PermissionError, match="one-shot"):
        initialized._migrate_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
        )


def test_deployment_roundtrip_includes_temporal_state_but_no_history_buffer() -> None:
    config = _dev_config()
    model = SharedObservableCameraRayJepaV5MultiresTemporalV1(config).eval()
    with torch.no_grad():
        values = torch.linspace(
            -0.01,
            0.01,
            model.temporal_residual.output_projection.weight.numel(),
        ).reshape_as(model.temporal_residual.output_projection.weight)
        model.temporal_residual.output_projection.weight.copy_(values)
    deployment = model.deployment_state_dict()
    temporal_names = {
        name for name in deployment
        if name.startswith("evidence_head.temporal_residual.")
    }
    assert temporal_names == {
        "evidence_head.temporal_residual.input_projection.weight",
        "evidence_head.temporal_residual.normalization.weight",
        "evidence_head.temporal_residual.normalization.bias",
        "evidence_head.temporal_residual.spatial_projection.weight",
        "evidence_head.temporal_residual.output_projection.weight",
    }
    assert not any("history" in name for name in deployment)

    previous_image, current_image = _images()
    calibration = _calibration()
    with torch.no_grad():
        warm_before = model.forward_temporal_frame(
            previous_image,
            current_image,
            *calibration,
            torch.ones(1, dtype=torch.bool),
        )
        cold_before = model.forward_temporal_frame(
            previous_image,
            current_image,
            *calibration,
            torch.zeros(1, dtype=torch.bool),
        )

    restored = SharedObservableCameraRayJepaV5MultiresTemporalV1(config).eval()
    restored.load_deployment_state_dict(deployment)
    assert tensor_state_dict_sha256(restored.deployment_state_dict()) == (
        tensor_state_dict_sha256(deployment)
    )
    with torch.no_grad():
        warm_after = restored.forward_temporal_frame(
            previous_image,
            current_image,
            *calibration,
            torch.ones(1, dtype=torch.bool),
        )
        cold_after = restored.forward_temporal_frame(
            previous_image,
            current_image,
            *calibration,
            torch.zeros(1, dtype=torch.bool),
        )
    assert torch.equal(warm_before.patch_tokens, warm_after.patch_tokens)
    assert torch.equal(warm_before.bev, warm_after.bev)
    _assert_evidence_equal(warm_before.evidence, warm_after.evidence)
    assert torch.equal(cold_before.patch_tokens, cold_after.patch_tokens)
    assert torch.equal(cold_before.bev, cold_after.bev)
    _assert_evidence_equal(cold_before.evidence, cold_after.evidence)


def test_temporal_block_synthetic_microfit_reduces_warm_target_error() -> None:
    student = VisualTokenDifferenceResidualV1()
    teacher = VisualTokenDifferenceResidualV1()
    with torch.no_grad():
        teacher.output_projection.weight.copy_(
            torch.linspace(
                -0.02,
                0.02,
                teacher.output_projection.weight.numel(),
            ).reshape_as(teacher.output_projection.weight)
        )
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    student.output_projection.weight.requires_grad_(True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(29)
    previous = torch.randn(2, 256, 192, generator=generator)
    current = torch.randn(2, 256, 192, generator=generator)
    valid = torch.ones(2, dtype=torch.bool)
    with torch.no_grad():
        target = teacher(previous, current, valid)
        initial = torch.nn.functional.mse_loss(
            student(previous, current, valid),
            target,
        )

    optimizer = torch.optim.Adam(
        (student.output_projection.weight,),
        lr=2e-2,
    )
    for _step in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(
            student(previous, current, valid),
            target,
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final = torch.nn.functional.mse_loss(
            student(previous, current, valid),
            target,
        )
        cold = student(
            previous,
            current,
            torch.zeros(2, dtype=torch.bool),
        )
    assert final < initial * 0.25
    assert torch.equal(cold, current)
