from __future__ import annotations

import inspect
import math

import pytest
import torch
import torch.nn as nn

from lewm.models.observable_camera_ray_evidence_v4 import (
    ENCODER_DIM,
    IMAGE_SIZE,
    TOKEN_SIDE,
    ObservableCameraRayEvidenceV4Model,
)
from lewm.models.shared_observable_camera_ray_jepa_v5 import (
    SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA,
    SharedObservableCameraRayJepaV5Config,
    tensor_state_dict_sha256,
)
from lewm.models.shared_observable_camera_ray_jepa_v5_multires_motion_alignment_v1 import (
    ALIGNMENT_INITIALIZATION_SEED,
    ALIGNMENT_WIDTH,
    BASIS_ORTHONORMAL_ATOL,
    EXPECTED_ALIGNMENT_PARAMETER_COUNT,
    EXPECTED_ALIGNMENT_PARAMETER_TENSOR_COUNT,
    EXPECTED_EVIDENCE_HEAD_PARAMETER_COUNT,
    EXPECTED_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT,
    EXPECTED_POST_ENCODER_PARAMETER_COUNT,
    EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT,
    EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT,
    MAXIMUM_OFFSET_TOKENS,
    MOTION_CONDITION_DIM,
    N320_CHECKPOINT_CONTENT_SHA256,
    N320_CHECKPOINT_FILE_SHA256,
    MotionConditionedTokenAlignmentV1,
    SharedObservableCameraRayJepaV5MultiresMotionAlignmentV1,
    build_causal_motion_condition_v1,
    motion_alignment_architecture_contract_v1,
)
from lewm.models.shared_observable_camera_ray_jepa_v5_multires_temporal_v1 import (
    EXPECTED_TEMPORAL_PARAMETER_COUNT,
    EXPECTED_TEMPORAL_PARAMETER_TENSOR_COUNT,
    SharedObservableCameraRayJepaV5MultiresTemporalV1,
)
from lewm.models.shared_observable_camera_ray_jepa_v5_multires_v1 import (
    EXPECTED_ENCODER_PARAMETER_COUNT,
    EXPECTED_ENCODER_PARAMETER_TENSOR_COUNT,
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


def _basis(
    roll: tuple[float, ...] | list[float],
    pitch: tuple[float, ...] | list[float],
) -> torch.Tensor:
    """Return stored FRU rows for ``Ry(pitch) @ Rx(roll)``."""

    roll_tensor = torch.tensor(roll, dtype=torch.float32)
    pitch_tensor = torch.tensor(pitch, dtype=torch.float32)
    if roll_tensor.shape != pitch_tensor.shape or roll_tensor.ndim != 1:
        raise ValueError("roll and pitch must be matching one-dimensional values")
    sine_roll = torch.sin(roll_tensor)
    cosine_roll = torch.cos(roll_tensor)
    sine_pitch = torch.sin(pitch_tensor)
    cosine_pitch = torch.cos(pitch_tensor)
    zeros = torch.zeros_like(roll_tensor)
    forward = torch.stack((cosine_pitch, zeros, -sine_pitch), dim=1)
    left = torch.stack(
        (
            sine_pitch * sine_roll,
            cosine_roll,
            cosine_pitch * sine_roll,
        ),
        dim=1,
    )
    up = torch.stack(
        (
            sine_pitch * cosine_roll,
            -sine_roll,
            cosine_pitch * cosine_roll,
        ),
        dim=1,
    )
    # The repository stores [forward, right, up], where right == -left.
    return torch.stack((forward, -left, up), dim=1)


def _calibration(batch: int = 1) -> tuple[torch.Tensor, ...]:
    origin = torch.tensor((0.326, 0.02, 0.043))[None].expand(batch, -1).clone()
    basis = _basis([0.0] * batch, [0.0] * batch)
    ground = torch.full((batch,), -0.35)
    return origin, basis, ground


def _images(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260726)
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


def _fit_model(
    config: SharedObservableCameraRayJepaV5Config,
) -> ObservableCameraRayEvidenceV4Model:
    return ObservableCameraRayEvidenceV4Model(
        encoder_depth=config.encoder_depth,
        source_shape=config.source_shape,
        pixel_ray_shape=config.pixel_ray_shape,
        query_chunk_size=config.query_chunk_size,
    )


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


def _tokens(
    batch: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(727)
    previous = torch.randn(
        batch,
        TOKEN_SIDE * TOKEN_SIDE,
        ENCODER_DIM,
        generator=generator,
    )
    current = torch.randn(
        batch,
        TOKEN_SIDE * TOKEN_SIDE,
        ENCODER_DIM,
        generator=generator,
    )
    condition = torch.randn(
        batch,
        MOTION_CONDITION_DIM,
        generator=generator,
    )
    return previous, current, condition


def test_alignment_topology_initialization_rng_and_capacity_are_exact() -> None:
    torch.manual_seed(31)
    caller_rng = torch.random.get_rng_state().clone()
    first = MotionConditionedTokenAlignmentV1()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    torch.manual_seed(999)
    second_caller_rng = torch.random.get_rng_state().clone()
    second = MotionConditionedTokenAlignmentV1()
    assert torch.equal(torch.random.get_rng_state(), second_caller_rng)
    assert tensor_state_dict_sha256(first.state_dict()) == (
        tensor_state_dict_sha256(second.state_dict())
    )

    assert tuple(first.state_dict()) == (
        "input_projection.weight",
        "input_projection.bias",
        "spatial_projection.weight",
        "offset_projection.weight",
    )
    assert "identity_grid_xy" not in first.state_dict()
    assert dict(first.named_buffers())["identity_grid_xy"].shape == (
        1,
        TOKEN_SIDE,
        TOKEN_SIDE,
        2,
    )
    assert (
        first.input_projection.in_channels,
        first.input_projection.out_channels,
        first.input_projection.kernel_size,
        first.input_projection.bias is not None,
    ) == (389, ALIGNMENT_WIDTH, (1, 1), True)
    assert (
        first.spatial_projection.in_channels,
        first.spatial_projection.out_channels,
        first.spatial_projection.kernel_size,
        first.spatial_projection.groups,
        first.spatial_projection.bias,
    ) == (ALIGNMENT_WIDTH, ALIGNMENT_WIDTH, (3, 3), ALIGNMENT_WIDTH, None)
    assert (
        first.offset_projection.in_channels,
        first.offset_projection.out_channels,
        first.offset_projection.kernel_size,
        first.offset_projection.bias,
    ) == (ALIGNMENT_WIDTH, 2, (1, 1), None)
    assert first.activation.approximate == "none"
    assert torch.count_nonzero(first.input_projection.weight).item() > 0
    assert torch.count_nonzero(first.input_projection.bias).item() == 0
    assert torch.count_nonzero(first.spatial_projection.weight).item() > 0
    assert torch.count_nonzero(first.offset_projection.weight).item() == 0
    assert _parameter_count(first) == EXPECTED_ALIGNMENT_PARAMETER_COUNT
    assert (
        _parameter_tensor_count(first)
        == EXPECTED_ALIGNMENT_PARAMETER_TENSOR_COUNT
    )


def test_warp_grid_sign_axis_order_border_and_identity_are_exact() -> None:
    block = MotionConditionedTokenAlignmentV1()
    rows = torch.arange(TOKEN_SIDE, dtype=torch.float32)[:, None].expand(
        TOKEN_SIDE, TOKEN_SIDE
    )
    columns = torch.arange(TOKEN_SIDE, dtype=torch.float32)[None].expand(
        TOKEN_SIDE, TOKEN_SIDE
    )
    token_map = torch.zeros(TOKEN_SIDE, TOKEN_SIDE, ENCODER_DIM)
    token_map[:, :, 0] = columns
    token_map[:, :, 1] = rows
    tokens = token_map.reshape(1, TOKEN_SIDE * TOKEN_SIDE, ENCODER_DIM)

    zero = torch.zeros(1, 2, TOKEN_SIDE, TOKEN_SIDE)
    identity = block.warp_previous_tokens(tokens, zero)
    assert torch.allclose(identity, tokens, rtol=0.0, atol=2e-5)

    positive_x = zero.clone()
    positive_x[:, 0] = 1.0
    warped_x = block.warp_previous_tokens(tokens, positive_x).reshape(
        TOKEN_SIDE, TOKEN_SIDE, ENCODER_DIM
    )
    assert torch.allclose(
        warped_x[:, :-1, 0],
        columns[:, 1:],
        rtol=0.0,
        atol=2e-5,
    )
    assert torch.equal(
        warped_x[:, -1, 0],
        torch.full((TOKEN_SIDE,), float(TOKEN_SIDE - 1)),
    )
    assert torch.allclose(warped_x[:, :, 1], rows, rtol=0.0, atol=2e-5)

    positive_y = zero.clone()
    positive_y[:, 1] = 1.0
    warped_y = block.warp_previous_tokens(tokens, positive_y).reshape(
        TOKEN_SIDE, TOKEN_SIDE, ENCODER_DIM
    )
    assert torch.allclose(
        warped_y[:-1, :, 1],
        rows[1:, :],
        rtol=0.0,
        atol=2e-5,
    )
    assert torch.equal(
        warped_y[-1, :, 1],
        torch.full((TOKEN_SIDE,), float(TOKEN_SIDE - 1)),
    )
    assert torch.allclose(warped_y[:, :, 0], columns, rtol=0.0, atol=2e-5)


def test_condition_builder_validates_fru_wraps_angles_and_zeros_cold_rows() -> None:
    previous = _basis(
        [math.pi - 0.1, 0.2],
        [0.1, -0.2],
    )
    current = _basis(
        [-math.pi + 0.1, 0.3],
        [0.4, -0.1],
    )
    nominal = torch.tensor(((0.7, -0.2, 0.3), (9.0, 8.0, 7.0)))
    history_valid = torch.tensor((True, False))
    condition = build_causal_motion_condition_v1(
        previous,
        current,
        nominal,
        history_valid,
    )
    assert condition.shape == (2, MOTION_CONDITION_DIM)
    assert torch.allclose(
        condition[0],
        torch.tensor((0.7, -0.2, 0.3, 0.2, 0.3)),
        rtol=0.0,
        atol=2e-5,
    )
    assert torch.equal(condition[1], torch.zeros(MOTION_CONDITION_DIM))

    reflected = current.clone()
    reflected[0, 1].neg_()
    with pytest.raises(ValueError, match="FRU|determinant"):
        build_causal_motion_condition_v1(
            previous,
            reflected,
            nominal,
            history_valid,
        )
    nonorthonormal = current.clone()
    nonorthonormal[0, 0, 0] += 2.0 * BASIS_ORTHONORMAL_ATOL
    with pytest.raises(ValueError, match="orthonormal"):
        build_causal_motion_condition_v1(
            previous,
            nonorthonormal,
            nominal,
            history_valid,
        )
    nonfinite = nominal.clone()
    nonfinite[0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        build_causal_motion_condition_v1(
            previous,
            current,
            nonfinite,
            history_valid,
        )
    with pytest.raises(ValueError, match="shape"):
        build_causal_motion_condition_v1(
            previous,
            current,
            nominal[:, :2],
            history_valid,
        )


def test_offsets_are_bounded_validate_inputs_and_cold_bypasses_exactly() -> None:
    block = MotionConditionedTokenAlignmentV1()
    with torch.no_grad():
        block.offset_projection.weight.fill_(10.0)
    previous, current, condition = _tokens()
    history_valid = torch.tensor((True, False))
    offset = block.predict_offset_tokens(
        previous,
        current,
        condition,
        history_valid,
    )
    assert torch.isfinite(offset).all()
    assert bool((offset.abs() <= MAXIMUM_OFFSET_TOKENS).all())
    assert torch.equal(offset[1], torch.zeros_like(offset[1]))
    aligned = block(previous, current, condition, history_valid)
    assert torch.equal(aligned[1], previous[1])

    invalid = condition.clone()
    invalid[0, 0] = torch.inf
    with pytest.raises(FloatingPointError, match="nonfinite"):
        block(previous, current, invalid, history_valid)
    with pytest.raises(ValueError, match="shape"):
        block(
            previous[:, :-1],
            current[:, :-1],
            condition,
            history_valid,
        )


def test_production_capacity_partition_contract_and_public_apis_are_exact() -> None:
    model = SharedObservableCameraRayJepaV5MultiresMotionAlignmentV1()
    assert _parameter_count(model.motion_alignment) == (
        EXPECTED_ALIGNMENT_PARAMETER_COUNT
    )
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
    assert sum(parameter.numel() for _name, parameter in trainable) == (
        EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT
    )
    assert len(trainable) == EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT
    assert all(
        name.startswith(("encoder.", "evidence_head."))
        for name, _parameter in trainable
    )
    assert (
        EXPECTED_TEMPORAL_PARAMETER_COUNT + EXPECTED_ALIGNMENT_PARAMETER_COUNT
        == EXPECTED_POST_ENCODER_PARAMETER_COUNT
    )

    contract = motion_alignment_architecture_contract_v1()
    assert contract["alignment"]["input_channels"] == 389
    assert contract["alignment"]["initialization_seed"] == (
        ALIGNMENT_INITIALIZATION_SEED
    )
    assert contract["alignment"]["source_grid"] == "identity_plus_offset_xy"
    assert contract["alignment"]["grid_sample_align_corners"] is True
    assert contract["condition"]["primitive_id_input"] is False
    assert contract["condition"]["realized_se2_input"] is False
    assert contract["jepa_objective_count"] == 0

    forbidden = {"primitive_id", "primitive_ids", "realized_se2", "realized_delta"}
    for method in (
        model.forward_camera_pair,
        model.forward_temporal_frame,
        model.build_causal_motion_condition,
    ):
        assert forbidden.isdisjoint(inspect.signature(method).parameters)
    with pytest.raises(PermissionError, match="bypasses temporal fusion"):
        model.forward_training_pair()


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
    aligned, receipt = (
        SharedObservableCameraRayJepaV5MultiresMotionAlignmentV1
        .initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=N320_CHECKPOINT_FILE_SHA256,
            n320_checkpoint_content_sha256=N320_CHECKPOINT_CONTENT_SHA256,
            config=config,
        )
    )
    predecessor.eval()
    aligned.eval()

    predecessor_state = predecessor.state_dict()
    aligned_state = aligned.state_dict()
    assert all(
        name in aligned_state and torch.equal(value, aligned_state[name])
        for name, value in predecessor_state.items()
    )
    assert set(aligned_state) - set(predecessor_state) == {
        "evidence_head.temporal_residual.input_projection.weight",
        "evidence_head.temporal_residual.normalization.weight",
        "evidence_head.temporal_residual.normalization.bias",
        "evidence_head.temporal_residual.spatial_projection.weight",
        "evidence_head.temporal_residual.output_projection.weight",
        "evidence_head.motion_alignment.input_projection.weight",
        "evidence_head.motion_alignment.input_projection.bias",
        "evidence_head.motion_alignment.spatial_projection.weight",
        "evidence_head.motion_alignment.offset_projection.weight",
    }

    previous_image, current_image = _images()
    calibration = _calibration()
    with torch.no_grad():
        expected_previous = predecessor.forward_frame(previous_image, *calibration)
        expected_current = predecessor.forward_frame(current_image, *calibration)
        observed_previous, observed_current = aligned.forward_camera_pair(
            previous_image,
            current_image,
            *calibration,
            *calibration,
            nominal_delta_current_frame=torch.zeros(1, 3),
        )
    assert torch.equal(observed_previous.patch_tokens, expected_previous.patch_tokens)
    assert torch.equal(observed_previous.bev, expected_previous.bev)
    _assert_evidence_equal(observed_previous.evidence, expected_previous.evidence)
    assert torch.equal(observed_current.patch_tokens, expected_current.patch_tokens)
    assert torch.equal(observed_current.bev, expected_current.bev)
    _assert_evidence_equal(observed_current.evidence, expected_current.evidence)
    assert receipt.alignment_initialization_seed == ALIGNMENT_INITIALIZATION_SEED
    assert receipt.alignment_state_sha256 == tensor_state_dict_sha256(
        aligned.motion_alignment.state_dict()
    )
    assert receipt.copied_alignment_entry_count == 0
    assert receipt.alignment_offset_projection_exact_zero is True
    assert receipt.copied_temporal_entry_count == 0
    assert receipt.temporal_output_projection_exact_zero is True


def test_learned_alignment_and_temporal_state_keep_cold_path_exact() -> None:
    model = SharedObservableCameraRayJepaV5MultiresMotionAlignmentV1(
        _dev_config()
    ).eval()
    with torch.no_grad():
        model.temporal_residual.output_projection.weight.fill_(0.01)
        model.motion_alignment.offset_projection.weight.fill_(0.01)
    previous_a, current = _images()
    previous_b = -previous_a
    origin, basis, ground = _calibration()
    cold = torch.zeros(1, dtype=torch.bool)
    warm = torch.ones(1, dtype=torch.bool)
    with torch.no_grad():
        baseline = model.forward_frame(current, origin, basis, ground)
        cold_a = model.forward_temporal_frame(
            previous_image=previous_a,
            current_image=current,
            previous_camera_basis_body_fru=basis,
            target_camera_origin_body_m=origin,
            target_camera_basis_body_fru=basis,
            target_ground_plane_z_body_m=ground,
            nominal_delta_current_frame=torch.tensor(((4.0, -3.0, 2.0),)),
            history_valid=cold,
        )
        cold_b = model.forward_temporal_frame(
            previous_image=previous_b,
            current_image=current,
            previous_camera_basis_body_fru=basis,
            target_camera_origin_body_m=origin,
            target_camera_basis_body_fru=basis,
            target_ground_plane_z_body_m=ground,
            nominal_delta_current_frame=torch.tensor(((-8.0, 7.0, -6.0),)),
            history_valid=cold,
        )
        warm_a = model.forward_temporal_frame(
            previous_image=previous_a,
            current_image=current,
            previous_camera_basis_body_fru=basis,
            target_camera_origin_body_m=origin,
            target_camera_basis_body_fru=basis,
            target_ground_plane_z_body_m=ground,
            nominal_delta_current_frame=torch.tensor(((0.1, 0.0, 0.0),)),
            history_valid=warm,
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


def test_three_stage_gradient_flow_reaches_both_images_condition_and_alignment() -> None:
    model = SharedObservableCameraRayJepaV5MultiresMotionAlignmentV1(
        _dev_config(encoder_depth=1)
    ).train()
    optimized = list(model.temporal_residual.parameters()) + list(
        model.motion_alignment.parameters()
    )
    optimizer = torch.optim.SGD(optimized, lr=2e-3)
    previous_seed, current_seed = _images()
    basis = _calibration()[1]
    history_valid = torch.ones(1, dtype=torch.bool)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(717)
    probe = torch.randn(
        1,
        TOKEN_SIDE * TOKEN_SIDE,
        ENCODER_DIM,
        generator=generator,
    )

    def backward_stage() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        previous = previous_seed.clone().requires_grad_(True)
        current = current_seed.clone().requires_grad_(True)
        nominal = torch.tensor(
            ((0.2, -0.1, 0.05),),
            requires_grad=True,
        )
        previous_tokens, current_tokens = model._encode_pair(previous, current)
        fused = model._fuse_motion_aligned_tokens(
            previous_tokens,
            current_tokens,
            basis,
            basis,
            nominal,
            history_valid,
        )
        (fused * probe).mean().backward()
        return previous, current, nominal

    previous_first, _current_first, nominal_first = backward_stage()
    assert torch.count_nonzero(
        model.temporal_residual.output_projection.weight.grad
    ).item() > 0
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() == 0
        for parameter in model.motion_alignment.parameters()
    )
    assert torch.count_nonzero(previous_first.grad).item() == 0
    assert torch.count_nonzero(nominal_first.grad).item() == 0
    optimizer.step()

    _previous_second, _current_second, _nominal_second = backward_stage()
    assert torch.count_nonzero(
        model.motion_alignment.offset_projection.weight.grad
    ).item() > 0
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() == 0
        for parameter in (
            model.motion_alignment.input_projection.weight,
            model.motion_alignment.input_projection.bias,
            model.motion_alignment.spatial_projection.weight,
        )
    )
    optimizer.step()

    previous_third, current_third, nominal_third = backward_stage()
    assert previous_third.grad is not None
    assert current_third.grad is not None
    assert nominal_third.grad is not None
    assert torch.isfinite(previous_third.grad).all()
    assert torch.isfinite(current_third.grad).all()
    assert torch.isfinite(nominal_third.grad).all()
    assert torch.count_nonzero(previous_third.grad).item() > 0
    assert torch.count_nonzero(current_third.grad).item() > 0
    assert torch.count_nonzero(nominal_third.grad).item() > 0
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.motion_alignment.parameters()
    )


def test_n320_receipt_copies_only_84_frozen_allowlisted_entries() -> None:
    torch.manual_seed(73)
    fit = ObservableCameraRayEvidenceV4Model()
    torch.manual_seed(918)
    caller_rng = torch.random.get_rng_state().clone()
    model, receipt = (
        SharedObservableCameraRayJepaV5MultiresMotionAlignmentV1
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
    assert receipt.copied_alignment_entry_count == 0
    assert not any(
        "dense_decoder" in name
        or "temporal_residual" in name
        or "motion_alignment" in name
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
    assert receipt.alignment_offset_projection_exact_zero is True
    assert tensor_state_dict_sha256(model.encoder.state_dict()) == (
        tensor_state_dict_sha256(model.target_encoder.state_dict())
    )
    assert receipt.alignment_state_sha256 == tensor_state_dict_sha256(
        model.motion_alignment.state_dict()
    )


def test_deployment_roundtrip_includes_all_learned_state_and_no_identity_grid() -> None:
    config = _dev_config()
    model = SharedObservableCameraRayJepaV5MultiresMotionAlignmentV1(config).eval()
    with torch.no_grad():
        model.temporal_residual.output_projection.weight.copy_(
            torch.linspace(
                -0.01,
                0.01,
                model.temporal_residual.output_projection.weight.numel(),
            ).reshape_as(model.temporal_residual.output_projection.weight)
        )
        model.motion_alignment.offset_projection.weight.copy_(
            torch.linspace(
                -0.005,
                0.005,
                model.motion_alignment.offset_projection.weight.numel(),
            ).reshape_as(model.motion_alignment.offset_projection.weight)
        )
    deployment = model.deployment_state_dict()
    alignment_names = {
        name
        for name in deployment
        if name.startswith("evidence_head.motion_alignment.")
    }
    assert alignment_names == {
        "evidence_head.motion_alignment.input_projection.weight",
        "evidence_head.motion_alignment.input_projection.bias",
        "evidence_head.motion_alignment.spatial_projection.weight",
        "evidence_head.motion_alignment.offset_projection.weight",
    }
    assert not any(
        "history" in name or "identity_grid" in name for name in deployment
    )

    previous, current, _condition = _tokens(batch=1)
    basis = _calibration()[1]
    nominal = torch.tensor(((0.2, -0.1, 0.05),))
    with torch.no_grad():
        warm_before = model._fuse_motion_aligned_tokens(
            previous,
            current,
            basis,
            basis,
            nominal,
            torch.ones(1, dtype=torch.bool),
        )
        cold_before = model._fuse_motion_aligned_tokens(
            previous,
            current,
            basis,
            basis,
            nominal,
            torch.zeros(1, dtype=torch.bool),
        )

    restored = SharedObservableCameraRayJepaV5MultiresMotionAlignmentV1(
        config
    ).eval()
    restored.load_deployment_state_dict(deployment)
    assert tensor_state_dict_sha256(restored.deployment_state_dict()) == (
        tensor_state_dict_sha256(deployment)
    )
    with torch.no_grad():
        warm_after = restored._fuse_motion_aligned_tokens(
            previous,
            current,
            basis,
            basis,
            nominal,
            torch.ones(1, dtype=torch.bool),
        )
        cold_after = restored._fuse_motion_aligned_tokens(
            previous,
            current,
            basis,
            basis,
            nominal,
            torch.zeros(1, dtype=torch.bool),
        )
    assert torch.equal(warm_before, warm_after)
    assert torch.equal(cold_before, cold_after)
    assert torch.equal(cold_after, current)


def test_alignment_block_cpu_microfit_reduces_warm_offset_error() -> None:
    student = MotionConditionedTokenAlignmentV1()
    teacher = MotionConditionedTokenAlignmentV1()
    with torch.no_grad():
        teacher.offset_projection.weight.copy_(
            torch.linspace(
                -0.003,
                0.003,
                teacher.offset_projection.weight.numel(),
            ).reshape_as(teacher.offset_projection.weight)
        )
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    student.offset_projection.weight.requires_grad_(True)

    previous, current, condition = _tokens()
    warm = torch.ones(2, dtype=torch.bool)
    with torch.no_grad():
        target = teacher.predict_offset_tokens(
            previous,
            current,
            condition,
            warm,
        )
        initial = torch.nn.functional.mse_loss(
            student.predict_offset_tokens(
                previous,
                current,
                condition,
                warm,
            ),
            target,
        )

    optimizer = torch.optim.Adam(
        (student.offset_projection.weight,),
        lr=1e-3,
    )
    for _step in range(16):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(
            student.predict_offset_tokens(
                previous,
                current,
                condition,
                warm,
            ),
            target,
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final = torch.nn.functional.mse_loss(
            student.predict_offset_tokens(
                previous,
                current,
                condition,
                warm,
            ),
            target,
        )
        cold = student(
            previous,
            current,
            condition,
            torch.zeros(2, dtype=torch.bool),
        )
    assert initial > 0
    assert final < initial * 0.25
    assert torch.equal(cold, previous)


def test_motion_model_is_additive_on_frozen_temporal_v1() -> None:
    config = _dev_config()
    torch.manual_seed(121)
    temporal = SharedObservableCameraRayJepaV5MultiresTemporalV1(config)
    torch.manual_seed(121)
    aligned = SharedObservableCameraRayJepaV5MultiresMotionAlignmentV1(config)
    temporal_state = temporal.state_dict()
    aligned_state = aligned.state_dict()
    assert all(
        name in aligned_state and torch.equal(value, aligned_state[name])
        for name, value in temporal_state.items()
    )
    assert set(aligned_state) - set(temporal_state) == {
        "evidence_head.motion_alignment.input_projection.weight",
        "evidence_head.motion_alignment.input_projection.bias",
        "evidence_head.motion_alignment.spatial_projection.weight",
        "evidence_head.motion_alignment.offset_projection.weight",
    }
