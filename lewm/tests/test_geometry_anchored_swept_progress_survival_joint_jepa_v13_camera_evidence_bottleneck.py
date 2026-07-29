from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4Model,
    ObservableCameraRayEvidenceV4RawOutput,
    ordered_obstacle_first_hit_log_probabilities_v4,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck import (
    CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13,
    FREE_EVIDENCE_PLANE_COUNT_V13,
    NOMINAL_CAMERA_BASIS_BODY_FRU_V13,
    NOMINAL_CAMERA_ORIGIN_BODY_M_V13,
    NOMINAL_GROUND_PLANE_Z_BODY_M_V13,
    OCCUPIED_EVIDENCE_PLANE_COUNT_V13,
    ONLINE_TRAINABLE_PARAMETER_COUNT_V13,
    PREDICTOR_GROUP_PARAMETER_COUNT_V13,
    PROJECTION_INITIALIZATION_SEED_V13,
    REPRESENTATION_GROUP_PARAMETER_COUNT_V13,
    SHARED_ROUTE_PARAMETER_COUNT_V13,
    TARGET_BOTTLENECK_PARAMETER_COUNT_V13,
    CameraEvidenceBottleneckEncodingV13,
    GeometryAnchoredSweptProgressSurvivalJointJepaV13,
    neutral_disjoint_ternary_log_probabilities_v12,
    retained_occupied_evidence_planes_v13,
)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def n320_fit_model() -> ObservableCameraRayEvidenceV4Model:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(40_013)
        return ObservableCameraRayEvidenceV4Model().eval()
    finally:
        torch.random.set_rng_state(caller_rng)


@pytest.fixture(scope="module")
def v13_model(
    n320_fit_model: ObservableCameraRayEvidenceV4Model,
) -> GeometryAnchoredSweptProgressSurvivalJointJepaV13:
    caller_rng = torch.random.get_rng_state().clone()
    model = GeometryAnchoredSweptProgressSurvivalJointJepaV13(
        n320_fit_model,
        _sweep_masks(),
    ).eval()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    return model


def test_exact_migration_projection_initialization_and_parameter_accounting(
    n320_fit_model: ObservableCameraRayEvidenceV4Model,
    v13_model: GeometryAnchoredSweptProgressSurvivalJointJepaV13,
) -> None:
    source_head = {
        name: value
        for name, value in n320_fit_model.state_dict().items()
        if not name.startswith("encoder.")
    }
    migrated_head = v13_model.bev_lift.evidence_head.state_dict()
    assert migrated_head.keys() == source_head.keys()
    assert all(
        torch.equal(value, source_head[name])
        for name, value in migrated_head.items()
    )
    assert all(
        torch.equal(value, n320_fit_model.encoder.state_dict()[name])
        for name, value in v13_model.encoder.state_dict().items()
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(PROJECTION_INITIALIZATION_SEED_V13)
    expected_free = torch.empty_like(v13_model.bev_lift.free_projection.weight)
    nn.init.xavier_uniform_(expected_free, gain=1.0, generator=generator)
    expected_occupied = torch.empty_like(
        v13_model.bev_lift.occupied_projection.weight
    )
    nn.init.xavier_uniform_(expected_occupied, gain=1.0, generator=generator)
    assert torch.equal(v13_model.bev_lift.free_projection.weight, expected_free)
    assert torch.equal(
        v13_model.bev_lift.occupied_projection.weight,
        expected_occupied,
    )
    assert torch.count_nonzero(v13_model.bev_lift.free_projection.bias) == 0
    assert torch.count_nonzero(v13_model.bev_lift.occupied_projection.bias) == 0
    assert sum(
        parameter.numel()
        for module in (
            v13_model.bev_lift.free_projection,
            v13_model.bev_lift.occupied_projection,
        )
        for parameter in module.parameters()
    ) == CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13

    groups = v13_model.trainable_parameter_groups_v13()
    assert tuple(
        sum(parameter.numel() for _, parameter in group) for group in groups
    ) == (
        SHARED_ROUTE_PARAMETER_COUNT_V13,
        REPRESENTATION_GROUP_PARAMETER_COUNT_V13,
        PREDICTOR_GROUP_PARAMETER_COUNT_V13,
    )
    assert sum(
        parameter.numel()
        for parameter in v13_model.parameters()
        if parameter.requires_grad
    ) == ONLINE_TRAINABLE_PARAMETER_COUNT_V13
    assert sum(
        parameter.numel()
        for module in v13_model.target_modules()
        for parameter in module.parameters()
    ) == TARGET_BOTTLENECK_PARAMETER_COUNT_V13

    assert torch.equal(
        v13_model.bev_lift.nominal_camera_origin_body_m,
        torch.tensor(NOMINAL_CAMERA_ORIGIN_BODY_M_V13),
    )
    assert torch.equal(
        v13_model.bev_lift.nominal_camera_basis_body_fru,
        torch.tensor(NOMINAL_CAMERA_BASIS_BODY_FRU_V13),
    )
    assert v13_model.bev_lift.nominal_ground_plane_z_body_m.item() == pytest.approx(
        NOMINAL_GROUND_PLANE_Z_BODY_M_V13
    )
    assert v13_model.bev_lift.free_cell_valid_mask.dtype == torch.bool
    assert v13_model.bev_lift.occupied_cell_valid_mask.dtype == torch.bool
    assert int(v13_model.bev_lift.free_cell_valid_mask.sum()) == 2_024
    assert int(v13_model.bev_lift.occupied_cell_valid_mask.sum()) == 2_118

    for online, target in zip(
        v13_model.online_target_modules(),
        v13_model.target_modules(),
        strict=True,
    ):
        assert online.state_dict().keys() == target.state_dict().keys()
        assert all(
            torch.equal(value, target.state_dict()[name])
            for name, value in online.state_dict().items()
        )
        assert not target.training
        assert all(not parameter.requires_grad for parameter in target.parameters())
    assert int(v13_model.target_hard_sync_count.item()) == 1
    assert int(v13_model.ema_update_count.item()) == 0


def test_free_planes_preserve_subcell_support_order_and_validity_bits(
    v13_model: GeometryAnchoredSweptProgressSurvivalJointJepaV13,
) -> None:
    ground = torch.arange(128 * 128 * 5, dtype=torch.float32).reshape(
        1, 128, 128, 5
    )
    valid = torch.ones_like(ground, dtype=torch.bool)
    valid[0, 7, 8, 2] = False
    raw = ObservableCameraRayEvidenceV4RawOutput(
        pixel_first_hit_hazard_logits=torch.zeros((1, 64, 1, 1)),
        pixel_within_bin_offset_m=torch.zeros((1, 64, 1, 1)),
        ground_clear_to_target_logits=ground,
        ground_query_in_frustum=valid,
        ground_query_uv_px=torch.zeros((1, 128, 128, 5, 2)),
        ground_target_distance_m=torch.zeros((1, 128, 128, 5)),
    )
    planes = v13_model.bev_lift.free_evidence_planes(raw)
    assert tuple(planes.shape) == (1, FREE_EVIDENCE_PLANE_COUNT_V13, 64, 64)

    output_row, output_column = 3, 4
    expected = torch.stack(
        tuple(
            ground[0, 2 * output_row + row_delta, 2 * output_column + column_delta, support]
            for row_delta in (0, 1)
            for column_delta in (0, 1)
            for support in range(5)
        )
    )
    expected[12] = 0.0
    assert torch.equal(planes[0, :20, output_row, output_column], expected)
    expected_bits = torch.ones(20)
    expected_bits[12] = 0.0
    assert torch.equal(planes[0, 20:, output_row, output_column], expected_bits)

    nominal_query = v13_model.bev_lift._nominal_ground_query(
        batch=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    expected_mask = (
        v13_model.bev_lift._flatten_ground_supports(nominal_query.in_frustum)
        .squeeze(0)
        .any(dim=0)
    )
    assert torch.equal(v13_model.bev_lift.free_cell_valid_mask, expected_mask)


def test_retained_splat_conserves_one_ray_hit_and_keeps_depths_separate() -> None:
    hazard = torch.full((1, 64, 84, 112), -100.0)
    selected = (0, 7, 42, 56)
    hazard[selected] = 0.0
    hazard.requires_grad_(True)
    offset = torch.zeros_like(hazard, requires_grad=True)
    origin = torch.tensor([NOMINAL_CAMERA_ORIGIN_BODY_M_V13])
    basis = torch.tensor([NOMINAL_CAMERA_BASIS_BODY_FRU_V13])

    planes = retained_occupied_evidence_planes_v13(
        hazard,
        offset,
        origin,
        basis,
    )
    assert tuple(planes.shape) == (1, OCCUPIED_EVIDENCE_PLANE_COUNT_V13, 64, 64)
    expected_hit = ordered_obstacle_first_hit_log_probabilities_v4(hazard).hit[
        selected
    ].exp()
    torch.testing.assert_close(
        planes[0, 7].sum(),
        expected_hit,
        rtol=0.0,
        atol=1e-6,
    )
    assert planes[0, 6].max() < 1e-20
    assert planes[0, 7].sum() > 0.49
    assert planes[0, 8].max() < 1e-20
    assert bool(torch.isfinite(planes).all())

    planes[0, 7].square().sum().backward()
    assert hazard.grad is not None and bool(torch.isfinite(hazard.grad).all())
    assert offset.grad is not None and bool(torch.isfinite(offset.grad).all())
    with pytest.raises(ValueError, match="exact 84x112"):
        retained_occupied_evidence_planes_v13(
            torch.zeros((1, 64, 1, 1)),
            torch.zeros((1, 64, 1, 1)),
            origin,
            basis,
        )


def test_nominal_and_auxiliary_paths_are_isolated_and_decode_once(
    v13_model: GeometryAnchoredSweptProgressSurvivalJointJepaV13,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb = torch.randn(
        (1, 3, 112, 112),
        generator=torch.Generator().manual_seed(79),
    )
    basis = torch.tensor([NOMINAL_CAMERA_BASIS_BODY_FRU_V13])
    ground = torch.tensor([NOMINAL_GROUND_PLANE_Z_BODY_M_V13])
    origin_a = torch.tensor([NOMINAL_CAMERA_ORIGIN_BODY_M_V13])
    origin_b = origin_a.clone()
    origin_b[:, 1] += 0.05

    decode_count = 0
    original_decode = v13_model.bev_lift.evidence_head.decode_dense_features

    def counted_decode(tokens: torch.Tensor) -> torch.Tensor:
        nonlocal decode_count
        decode_count += 1
        return original_decode(tokens)

    monkeypatch.setattr(
        v13_model.bev_lift.evidence_head,
        "decode_dense_features",
        counted_decode,
    )
    with torch.no_grad():
        first = v13_model.encode_online_with_auxiliary_evidence(
            rgb,
            camera_origin_body_m=origin_a,
            camera_basis_body_fru=basis,
            ground_plane_z_body_m=ground,
        )
        second = v13_model.encode_online_training(
            rgb,
            camera_origin_body_m=origin_b,
            camera_basis_body_fru=basis,
            ground_plane_z_body_m=ground,
        )
    assert decode_count == 2
    assert torch.equal(first.latent, second.latent)
    assert torch.equal(first.free_evidence_planes, second.free_evidence_planes)
    assert torch.equal(
        first.occupied_evidence_planes,
        second.occupied_evidence_planes,
    )
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

    monkeypatch.undo()
    with torch.no_grad():
        nominal = v13_model.encode_online_with_evidence(rgb)
        direct = v13_model.encode_online(rgb)
        target = v13_model.encode_target(rgb)
    assert isinstance(nominal, CameraEvidenceBottleneckEncodingV13)
    assert torch.equal(nominal.latent, direct)
    assert torch.equal(direct, target)
    assert not target.requires_grad
    with pytest.raises(RuntimeError, match="removed encode_online_with_sampling"):
        v13_model.encode_online_with_sampling(rgb)
    with pytest.raises(RuntimeError, match="removed encode_target_with_sampling"):
        v13_model.encode_target_with_sampling(rgb)


def test_v13_masks_drive_semantics_and_ema_updates_only_final_bottleneck(
    v13_model: GeometryAnchoredSweptProgressSurvivalJointJepaV13,
) -> None:
    latent = torch.randn(
        (1, 64, 64, 64),
        generator=torch.Generator().manual_seed(83),
    )
    logits = v13_model.semantic_logits_from_latent(latent)
    assert tuple(logits.shape) == (1, 3, 64, 64)
    invalid = ~v13_model.bev_lift.cell_valid_mask
    if bool(invalid.any()):
        assert torch.equal(
            logits[0, :, invalid],
            logits.new_tensor((0.0, -20.0, -20.0))[:, None].expand(
                -1, int(invalid.sum())
            ),
        )
    free_invalid = ~v13_model.bev_lift.free_cell_valid_mask
    occupied_invalid = ~v13_model.bev_lift.occupied_cell_valid_mask
    free_axis, occupied_axis = v13_model.semantic_head.evidence_logits(latent)
    free_axis = torch.where(
        free_invalid[None], torch.full_like(free_axis, -20.0), free_axis
    )
    occupied_axis = torch.where(
        occupied_invalid[None],
        torch.full_like(occupied_axis, -20.0),
        occupied_axis,
    )
    expected = neutral_disjoint_ternary_log_probabilities_v12(
        free_axis,
        occupied_axis,
    )
    expected = torch.where(
        v13_model.bev_lift.cell_valid_mask[None, None],
        expected,
        expected.new_tensor((0.0, -20.0, -20.0))[None, :, None, None],
    )
    assert torch.equal(logits, expected)

    model = copy.deepcopy(v13_model).train()
    online = model.bev_lift.free_projection.weight
    target = model.target_bev_lift.free_projection.weight
    before = target.detach().clone()
    with torch.no_grad():
        online.add_(1.0)
    model.update_target_ema_after_optimizer_step()
    torch.testing.assert_close(
        target,
        before * 0.996 + online.detach() * 0.004,
        rtol=0.0,
        atol=2e-7,
    )
    assert int(model.target_hard_sync_count.item()) == 1
    assert int(model.ema_update_count.item()) == 1
    assert not model.target_encoder.training
    assert not model.target_bev_lift.training
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for module in model.target_modules()
        for parameter in module.parameters()
    )
