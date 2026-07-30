from __future__ import annotations

import copy
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
import torch.nn as nn

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    CAMERA_HORIZONTAL_FOV_DEG,
    CAMERA_VERTICAL_FOV_DEG,
    OUTPUT_CELL_SIZE_M,
    OUTPUT_FORWARD_MIN_EDGE_M,
    OUTPUT_LEFT_MIN_EDGE_M,
    OUTPUT_SHAPE,
    PIXEL_RAY_SHAPE,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    DEPTH_BIN_COUNT,
    DEPTH_BIN_SIZE_M,
    DEPTH_NEAR_EDGE_M,
    ObservableCameraRayEvidenceV4Model,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck import (
    NOMINAL_CAMERA_BASIS_BODY_FRU_V13,
    NOMINAL_CAMERA_ORIGIN_BODY_M_V13,
    NOMINAL_GROUND_PLANE_Z_BODY_M_V13,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_object_space_height_volume import (
    FLATTENED_VOLUME_CHANNEL_COUNT_V18,
    HEIGHT_CENTRES_M_V18,
    HEIGHT_COUNT_V18,
    HEIGHT_NORMALIZATION_CENTRE_M_V18,
    HEIGHT_NORMALIZATION_HALF_RANGE_M_V18,
    OBJECT_SPACE_HEIGHT_VOLUME_PARAMETER_COUNT_V18,
    OBJECT_SPACE_HEIGHT_VOLUME_SEMANTIC_PARAMETER_COUNT_V18,
    ONLINE_TRAINABLE_PARAMETER_COUNT_V18,
    PREDICTOR_GROUP_PARAMETER_COUNT_V18,
    REPRESENTATION_GROUP_PARAMETER_COUNT_V18,
    SHARED_ROUTE_PARAMETER_COUNT_V18,
    TARGET_BOTTLENECK_PARAMETER_COUNT_V18,
    VOXEL_INPUT_CHANNEL_COUNT_V18,
    VOLUME_CHANNEL_COUNT_V18,
    VOLUME_INITIALIZATION_SEED_V18,
    GeometryAnchoredSweptProgressSurvivalJointJepaV18,
    ObjectSpaceHeightVolumeLiftV18,
    ObjectSpaceHeightVolumeSemanticDecoderV18,
    flatten_height_major_volume_v18,
    object_space_voxel_geometry_v18,
    ordered_ray_volume_source_v18,
    unflatten_height_major_volume_v18,
)
from scripts import run_go2_rgb_object_space_height_volume_joint_jepa_v18 as runner


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def n320_fit_model() -> ObservableCameraRayEvidenceV4Model:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(18_001)
        return ObservableCameraRayEvidenceV4Model().eval()
    finally:
        torch.random.set_rng_state(caller_rng)


@pytest.fixture(scope="module")
def v18_model(
    n320_fit_model: ObservableCameraRayEvidenceV4Model,
) -> GeometryAnchoredSweptProgressSurvivalJointJepaV18:
    caller_rng = torch.random.get_rng_state().clone()
    model = GeometryAnchoredSweptProgressSurvivalJointJepaV18(
        n320_fit_model,
        _sweep_masks(),
    ).eval()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    return model


def test_ordered_ray_source_has_exact_first_hit_survival_and_offset_channels() -> None:
    hazard = torch.zeros((1, DEPTH_BIN_COUNT, *PIXEL_RAY_SHAPE))
    offset = torch.full_like(hazard, 0.025)
    source = ordered_ray_volume_source_v18(hazard, offset)

    assert tuple(source.shape) == (1, 3, DEPTH_BIN_COUNT, *PIXEL_RAY_SHAPE)
    depth = torch.arange(DEPTH_BIN_COUNT, dtype=torch.float32)
    expected_hit = torch.pow(0.5, depth + 1.0)
    expected_clear = torch.pow(0.5, depth + 0.5)
    torch.testing.assert_close(
        source[0, 0, :, 17, 29], expected_hit, rtol=5e-6, atol=0.0
    )
    torch.testing.assert_close(
        source[0, 1, :, 17, 29], expected_clear, rtol=5e-6, atol=0.0
    )
    assert torch.equal(source[:, 2], torch.full_like(source[:, 2], 0.5))
    assert bool(torch.isfinite(source).all())


def test_exact_xyz_height_geometry_visibility_and_depth_bin_alignment(
    v18_model: GeometryAnchoredSweptProgressSurvivalJointJepaV18,
) -> None:
    geometry = object_space_voxel_geometry_v18()
    height = torch.tensor(HEIGHT_CENTRES_M_V18, dtype=torch.float32)
    forward = OUTPUT_FORWARD_MIN_EDGE_M + (
        torch.arange(OUTPUT_SHAPE[0], dtype=torch.float32) + 0.5
    ) * OUTPUT_CELL_SIZE_M
    left = OUTPUT_LEFT_MIN_EDGE_M + (
        torch.arange(OUTPUT_SHAPE[1], dtype=torch.float32) + 0.5
    ) * OUTPUT_CELL_SIZE_M
    z, x, y = torch.meshgrid(height, forward, left, indexing="ij")
    expected_xyz = torch.stack((x, y, z), dim=-1)
    assert torch.equal(geometry.voxel_xyz_body_m, expected_xyz)
    assert torch.equal(geometry.voxel_xyz_body_m[:, 0, 0, 2], height)
    assert torch.equal(
        geometry.voxel_xyz_body_m[0, :, 0, 0],
        forward,
    )
    assert torch.equal(geometry.voxel_xyz_body_m[0, 0, :, 1], left)

    origin = torch.tensor(NOMINAL_CAMERA_ORIGIN_BODY_M_V13)
    basis = torch.tensor(NOMINAL_CAMERA_BASIS_BODY_FRU_V13)
    delta = expected_xyz - origin
    camera_forward = torch.sum(delta * basis[0], dim=-1)
    camera_right = torch.sum(delta * basis[1], dim=-1)
    camera_up = torch.sum(delta * basis[2], dim=-1)
    range_m = torch.linalg.vector_norm(delta, dim=-1)
    grid_x = camera_right / (
        camera_forward * math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEG) / 2.0)
    )
    grid_y = -camera_up / (
        camera_forward * math.tan(math.radians(CAMERA_VERTICAL_FOV_DEG) / 2.0)
    )
    grid_z = 2.0 * (range_m - DEPTH_NEAR_EDGE_M) / (
        DEPTH_BIN_COUNT * DEPTH_BIN_SIZE_M
    ) - 1.0
    expected_grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)
    expected_visible = (
        (camera_forward >= 0.05)
        & (grid_x.abs() <= 1.0)
        & (grid_y.abs() <= 1.0)
        & (range_m >= 0.05)
        & (range_m <= 6.45)
    )
    assert torch.equal(geometry.sample_grid_xyz, expected_grid)
    assert torch.equal(geometry.voxel_visible, expected_visible)
    assert torch.equal(
        geometry.normalized_registered_height,
        (z - HEIGHT_NORMALIZATION_CENTRE_M_V18)
        / HEIGHT_NORMALIZATION_HALF_RANGE_M_V18,
    )
    assert torch.equal(
        v18_model.bev_lift.cell_valid_mask,
        expected_visible.any(dim=0),
    )

    # For align_corners=False, normalized z maps to this continuous bin index.
    expected_depth_index = (range_m - DEPTH_NEAR_EDGE_M) / DEPTH_BIN_SIZE_M - 0.5
    mapped_depth_index = (
        (geometry.sample_grid_xyz[..., 2] + 1.0) * DEPTH_BIN_COUNT - 1.0
    ) / 2.0
    torch.testing.assert_close(
        mapped_depth_index, expected_depth_index, rtol=2e-6, atol=2e-6
    )

    # Put the depth index itself in the normalized-offset channel and verify
    # that the lift's real trilinear sample lands on that continuous index.
    hazard = torch.zeros((1, DEPTH_BIN_COUNT, *PIXEL_RAY_SHAPE))
    depth_ramp = torch.arange(DEPTH_BIN_COUNT, dtype=torch.float32).reshape(
        1, DEPTH_BIN_COUNT, 1, 1
    )
    offset = (0.05 * depth_ramp).expand_as(hazard).contiguous()
    voxel_inputs, visible = v18_model.bev_lift.voxel_inputs_from_ray_field(
        hazard,
        offset,
    )
    fully_interior = (
        visible[0]
        & (grid_x.abs() < 1.0 - 1.0 / PIXEL_RAY_SHAPE[1])
        & (grid_y.abs() < 1.0 - 1.0 / PIXEL_RAY_SHAPE[0])
        & (grid_z.abs() < 1.0 - 1.0 / DEPTH_BIN_COUNT)
    )
    assert bool(fully_interior.any())
    torch.testing.assert_close(
        voxel_inputs[0, 2][fully_interior],
        expected_depth_index[fully_interior],
        rtol=1e-5,
        atol=2e-4,
    )


def test_height_major_flatten_order_and_inverse_are_exact() -> None:
    channel = torch.arange(VOLUME_CHANNEL_COUNT_V18).reshape(1, -1, 1, 1, 1)
    height = torch.arange(HEIGHT_COUNT_V18).reshape(1, 1, -1, 1, 1)
    volume = (100 * height + channel).expand(
        1,
        VOLUME_CHANNEL_COUNT_V18,
        HEIGHT_COUNT_V18,
        *OUTPUT_SHAPE,
    ).to(dtype=torch.float32)
    latent = flatten_height_major_volume_v18(volume)
    assert tuple(latent.shape) == (
        1,
        FLATTENED_VOLUME_CHANNEL_COUNT_V18,
        *OUTPUT_SHAPE,
    )
    for height_index in range(HEIGHT_COUNT_V18):
        for channel_index in range(VOLUME_CHANNEL_COUNT_V18):
            flattened_index = height_index * VOLUME_CHANNEL_COUNT_V18 + channel_index
            assert latent[0, flattened_index, 7, 11].item() == (
                100 * height_index + channel_index
            )
    assert torch.equal(unflatten_height_major_volume_v18(latent), volume)


def test_isolated_initialization_preserves_rng_and_exact_layer_order(
    n320_fit_model: ObservableCameraRayEvidenceV4Model,
) -> None:
    torch.random.default_generator.manual_seed(18_700)
    caller_rng = torch.random.get_rng_state().clone()
    lift = ObjectSpaceHeightVolumeLiftV18(n320_fit_model)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    semantic = ObjectSpaceHeightVolumeSemanticDecoderV18()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    generator = torch.Generator(device="cpu").manual_seed(
        VOLUME_INITIALIZATION_SEED_V18
    )
    for layer in (
        lift.point_projection,
        lift.volume_block.conv1,
        lift.volume_block.conv2,
    ):
        expected = torch.empty_like(layer.weight)
        nn.init.xavier_uniform_(expected, gain=1.0, generator=generator)
        assert torch.equal(layer.weight, expected)
        assert torch.count_nonzero(layer.bias) == 0

    generator = torch.Generator(device="cpu").manual_seed(
        VOLUME_INITIALIZATION_SEED_V18
    )
    for layer in (semantic.conv1, semantic.conv2, semantic.evidence_head):
        expected = torch.empty_like(layer.weight)
        nn.init.xavier_uniform_(expected, gain=1.0, generator=generator)
        assert torch.equal(layer.weight, expected)
        assert torch.count_nonzero(layer.bias) == 0
    lift_parameter_count = sum(
        parameter.numel() for parameter in lift.point_projection.parameters()
    ) + sum(parameter.numel() for parameter in lift.volume_block.parameters())
    assert lift_parameter_count == OBJECT_SPACE_HEIGHT_VOLUME_PARAMETER_COUNT_V18
    assert sum(parameter.numel() for parameter in semantic.parameters()) == (
        OBJECT_SPACE_HEIGHT_VOLUME_SEMANTIC_PARAMETER_COUNT_V18
    )


def test_output_shapes_masks_and_unknown_semantics(
    v18_model: GeometryAnchoredSweptProgressSurvivalJointJepaV18,
) -> None:
    tokens = torch.randn(
        (1, 256, 192),
        generator=torch.Generator().manual_seed(18_101),
    )
    with torch.no_grad():
        encoded = v18_model.bev_lift.forward_with_evidence(tokens)
        semantic = v18_model.semantic_logits_from_latent(encoded.latent)
    assert tuple(encoded.nominal_evidence.pixel_first_hit_hazard_logits.shape) == (
        1,
        DEPTH_BIN_COUNT,
        *PIXEL_RAY_SHAPE,
    )
    assert tuple(encoded.voxel_inputs.shape) == (
        1,
        VOXEL_INPUT_CHANNEL_COUNT_V18,
        HEIGHT_COUNT_V18,
        *OUTPUT_SHAPE,
    )
    assert tuple(encoded.voxel_visible.shape) == (
        1,
        HEIGHT_COUNT_V18,
        *OUTPUT_SHAPE,
    )
    assert tuple(encoded.height_volume.shape) == (
        1,
        VOLUME_CHANNEL_COUNT_V18,
        HEIGHT_COUNT_V18,
        *OUTPUT_SHAPE,
    )
    assert tuple(encoded.latent.shape) == (
        1,
        FLATTENED_VOLUME_CHANNEL_COUNT_V18,
        *OUTPUT_SHAPE,
    )
    assert torch.equal(
        encoded.voxel_visible,
        v18_model.bev_lift.voxel_visible_mask[None],
    )
    volume_mask = encoded.voxel_visible[:, None].expand_as(encoded.height_volume)
    input_mask = encoded.voxel_visible[:, None].expand_as(encoded.voxel_inputs)
    assert torch.count_nonzero(encoded.voxel_inputs[~input_mask]) == 0
    assert torch.count_nonzero(encoded.height_volume[~volume_mask]) == 0
    assert torch.equal(
        unflatten_height_major_volume_v18(encoded.latent),
        encoded.height_volume,
    )
    assert bool(torch.isfinite(encoded.latent).all())

    invalid = ~v18_model.bev_lift.cell_valid_mask
    assert bool(invalid.any())
    expected_unknown = semantic.new_tensor((0.0, -20.0, -20.0))[:, None].expand(
        -1, int(invalid.sum())
    )
    assert torch.equal(semantic[0, :, invalid], expected_unknown)


def test_auxiliary_calibration_cannot_change_nominal_volume(
    v18_model: GeometryAnchoredSweptProgressSurvivalJointJepaV18,
) -> None:
    tokens = torch.randn(
        (1, 256, 192),
        generator=torch.Generator().manual_seed(18_102),
    )
    origin_a = torch.tensor([NOMINAL_CAMERA_ORIGIN_BODY_M_V13])
    origin_b = origin_a.clone()
    origin_b[:, 1] += 0.10
    basis = torch.tensor([NOMINAL_CAMERA_BASIS_BODY_FRU_V13])
    ground = torch.tensor([NOMINAL_GROUND_PLANE_Z_BODY_M_V13])
    with torch.no_grad():
        first = v18_model.bev_lift.forward_with_auxiliary_evidence(
            tokens,
            camera_origin_body_m=origin_a,
            camera_basis_body_fru=basis,
            ground_plane_z_body_m=ground,
        )
        second = v18_model.bev_lift.forward_with_auxiliary_evidence(
            tokens,
            camera_origin_body_m=origin_b,
            camera_basis_body_fru=basis,
            ground_plane_z_body_m=ground,
        )
    assert torch.equal(first.latent, second.latent)
    assert torch.equal(first.voxel_inputs, second.voxel_inputs)
    assert torch.equal(first.height_volume, second.height_volume)
    for field in first.nominal_evidence.__dataclass_fields__:
        assert torch.equal(
            getattr(first.nominal_evidence, field),
            getattr(second.nominal_evidence, field),
        )
    assert not torch.equal(
        first.auxiliary_evidence.ground_query_uv_px,
        second.auxiliary_evidence.ground_query_uv_px,
    )
    assert first.nominal_evidence.pixel_first_hit_hazard_logits is (
        first.auxiliary_evidence.pixel_first_hit_hazard_logits
    )
    assert first.nominal_evidence.pixel_within_bin_offset_m is (
        first.auxiliary_evidence.pixel_within_bin_offset_m
    )


def test_semantic_and_predictor_gradients_reach_shared_pixel_head_through_volume(
    v18_model: GeometryAnchoredSweptProgressSurvivalJointJepaV18,
) -> None:
    v18_model.zero_grad(set_to_none=True)
    tokens = torch.randn(
        (1, 256, 192),
        generator=torch.Generator().manual_seed(18_103),
        requires_grad=True,
    )
    encoded = v18_model.bev_lift.forward_with_evidence(tokens)
    semantic = v18_model.semantic_head(encoded.latent)
    semantic_coefficients = torch.randn(
        semantic.shape,
        generator=torch.Generator().manual_seed(18_104),
    )
    semantic_loss = (semantic * semantic_coefficients).mean()

    actions = torch.eye(v18_model.config.action_dim, dtype=torch.float32)[2:3]
    predicted = v18_model.predict(encoded.latent, actions)
    prediction_coefficients = torch.randn(
        predicted.shape,
        generator=torch.Generator().manual_seed(18_105),
    )
    prediction_loss = (predicted * prediction_coefficients).mean()

    pixel_weight = v18_model.bev_lift.evidence_head.pixel_head.weight
    point_weight = v18_model.bev_lift.point_projection.weight
    semantic_weight = v18_model.semantic_head.evidence_head.weight
    semantic_gradients = torch.autograd.grad(
        semantic_loss,
        (pixel_weight, point_weight, semantic_weight),
        retain_graph=True,
    )
    predictor_parameters = tuple(v18_model.predictor.parameters())
    prediction_gradients = torch.autograd.grad(
        prediction_loss,
        (pixel_weight, point_weight, *predictor_parameters),
        allow_unused=True,
    )
    for name, gradient in zip(
        ("semantic pixel", "semantic point", "semantic head"),
        semantic_gradients,
        strict=True,
    ):
        assert bool(torch.isfinite(gradient).all())
        assert int(torch.count_nonzero(gradient)) > 0, name
    for name, gradient in zip(
        ("predictor pixel", "predictor point"),
        prediction_gradients[:2],
        strict=True,
    ):
        assert gradient is not None, name
        assert bool(torch.isfinite(gradient).all())
        assert int(torch.count_nonzero(gradient)) > 0, name
    used_predictor_gradients = tuple(
        gradient for gradient in prediction_gradients[2:] if gradient is not None
    )
    assert used_predictor_gradients
    assert all(
        bool(torch.isfinite(gradient).all())
        for gradient in used_predictor_gradients
    )
    assert any(
        int(torch.count_nonzero(gradient)) > 0
        for gradient in used_predictor_gradients
    )
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for module in v18_model.target_modules()
        for parameter in module.parameters()
    )


def test_parameter_groups_target_identity_and_exact_ema(
    v18_model: GeometryAnchoredSweptProgressSurvivalJointJepaV18,
) -> None:
    groups = v18_model.trainable_parameter_groups_v18()
    expected_counts = (
        SHARED_ROUTE_PARAMETER_COUNT_V18,
        REPRESENTATION_GROUP_PARAMETER_COUNT_V18,
        PREDICTOR_GROUP_PARAMETER_COUNT_V18,
    )
    assert tuple(
        sum(parameter.numel() for _, parameter in group) for group in groups
    ) == expected_counts
    selected = [(name, parameter) for group in groups for name, parameter in group]
    trainable = [
        (name, parameter)
        for name, parameter in v18_model.named_parameters()
        if parameter.requires_grad
    ]
    assert [name for name, _ in selected] == [name for name, _ in trainable]
    assert len({id(parameter) for _, parameter in selected}) == len(selected)
    assert sum(expected_counts) == ONLINE_TRAINABLE_PARAMETER_COUNT_V18
    assert sum(
        parameter.numel()
        for module in v18_model.target_modules()
        for parameter in module.parameters()
    ) == TARGET_BOTTLENECK_PARAMETER_COUNT_V18

    for online, target in zip(
        v18_model.online_target_modules(),
        v18_model.target_modules(),
        strict=True,
    ):
        assert online.state_dict().keys() == target.state_dict().keys()
        assert all(
            torch.equal(value, target.state_dict()[name])
            for name, value in online.state_dict().items()
        )
        assert not target.training
        assert all(not parameter.requires_grad for parameter in target.parameters())
    assert int(v18_model.target_hard_sync_count.item()) == 1
    assert int(v18_model.ema_update_count.item()) == 0

    model = copy.deepcopy(v18_model).train()
    online_weight = model.bev_lift.point_projection.weight
    target_weight = model.target_bev_lift.point_projection.weight
    before = target_weight.detach().clone()
    with torch.no_grad():
        online_weight.add_(0.125)
    model.update_target_ema_after_optimizer_step()
    expected = before * model.config.target_ema_momentum + online_weight.detach() * (
        1.0 - model.config.target_ema_momentum
    )
    torch.testing.assert_close(target_weight, expected, rtol=0.0, atol=2e-7)
    assert int(model.target_hard_sync_count.item()) == 1
    assert int(model.ema_update_count.item()) == 1
    assert not model.target_bev_lift.training
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for module in model.target_modules()
        for parameter in module.parameters()
    )


def test_one_real_synthetic_joint_update_has_all_routes_one_step_and_one_ema(
    v18_model: GeometryAnchoredSweptProgressSurvivalJointJepaV18,
) -> None:
    model = copy.deepcopy(v18_model).train()
    partition = runner.partition_parameters_v18(model)
    optimizer = runner.build_frozen_optimizer_v18(partition)
    batch_size = 4
    generator = torch.Generator().manual_seed(20_260_730)
    origin = torch.tensor(NOMINAL_CAMERA_ORIGIN_BODY_M_V13).expand(
        batch_size, -1
    ).clone()
    basis = torch.tensor(NOMINAL_CAMERA_BASIS_BODY_FRU_V13).expand(
        batch_size, -1, -1
    ).clone()
    ground = torch.full((batch_size,), NOMINAL_GROUND_PLANE_Z_BODY_M_V13)
    with torch.no_grad():
        query = model.bev_lift.evidence_head.ground_query_geometry(
            origin,
            basis,
            ground,
        )
    ground_valid = query.in_frustum.clone()
    ground_pattern = (
        torch.arange(128)[:, None, None]
        + torch.arange(128)[None, :, None]
        + torch.arange(5)[None, None, :]
    ) % 2 == 0
    ground_clear = (
        ground_pattern[None].expand(batch_size, -1, -1, -1) & ground_valid
    )
    hit = (
        (torch.arange(84)[:, None] + torch.arange(112)[None, :]) % 3 == 0
    )[None].expand(batch_size, -1, -1).clone()
    distance = torch.where(hit, torch.full(hit.shape, 1.25), torch.zeros(hit.shape))
    labels = (
        (torch.arange(64)[:, None] + torch.arange(64)[None, :]) % 3
    )[None].expand(batch_size, -1, -1).clone().long()
    actions = torch.tensor((0, 1, 2, 3), dtype=torch.long)
    feasible = torch.ones((batch_size, 9), dtype=torch.bool)
    prefix = torch.arange(1, 10, dtype=torch.long)[None].expand(
        batch_size, -1
    ).clone()
    microbatches = []
    for index in range(4):
        microbatches.append(
            {
                runner.CURRENT_RGB_KEY: torch.randn(
                    (batch_size, 3, 112, 112), generator=generator
                ),
                runner.NEXT_RGB_KEY: torch.randn(
                    (batch_size, 3, 112, 112), generator=generator
                ),
                runner.CURRENT_LABELS_KEY: labels.roll(index, dims=0).clone(),
                runner.NEXT_LABELS_KEY: labels.roll(index + 1, dims=0).clone(),
                runner.EXECUTED_ACTION_KEY: actions.clone(),
                runner.IMMEDIATE_FEASIBLE_KEY: feasible.clone(),
                runner.PREFIX_LENGTHS_KEY: prefix.clone(),
                runner.CURRENT_CAMERA_ORIGIN_KEY: origin.clone(),
                runner.NEXT_CAMERA_ORIGIN_KEY: origin.clone(),
                runner.CURRENT_CAMERA_BASIS_KEY: basis.clone(),
                runner.NEXT_CAMERA_BASIS_KEY: basis.clone(),
                runner.CURRENT_GROUND_PLANE_Z_KEY: ground.clone(),
                runner.NEXT_GROUND_PLANE_Z_KEY: ground.clone(),
                runner.CURRENT_PIXEL_HIT_KEY: hit.clone(),
                runner.NEXT_PIXEL_HIT_KEY: hit.roll(1, dims=2).clone(),
                runner.CURRENT_PIXEL_DISTANCE_KEY: distance.clone(),
                runner.NEXT_PIXEL_DISTANCE_KEY: distance.roll(1, dims=2).clone(),
                runner.CURRENT_GROUND_IN_FRUSTUM_KEY: ground_valid.clone(),
                runner.NEXT_GROUND_IN_FRUSTUM_KEY: ground_valid.clone(),
                runner.CURRENT_GROUND_CLEAR_KEY: ground_clear.clone(),
                runner.NEXT_GROUND_CLEAR_KEY: (~ground_clear & ground_valid).clone(),
            }
        )

    result = runner.joint_training_update_v18(
        model,
        optimizer,
        tuple(microbatches),
    )
    assert result.accounting == runner.JointTrainingAccountingV13(
        updates=1,
        presentations=16,
        microbatch_graphs=4,
        backward_calls=8,
        camera_route_grad_calls=4,
        joint_route_grad_calls=4,
        camera_frame_objectives=32,
        optimizer_steps=1,
        ema_steps=1,
        predictor_forwards=4,
        predictor_objectives=4,
    )
    assert all(
        receipt.preclip_l2 > 0.0 and receipt.absent_tensor_gradient_count == 0
        for receipt in result.gradient_routes.values()
    )
    assert result.optimizer_steps_this_update == 1
    assert result.ema_steps_this_update == 1
    assert result.target_gradient_tensor_count == 0
    assert int(model.ema_update_count.item()) == 1
