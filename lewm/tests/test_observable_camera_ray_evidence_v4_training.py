from __future__ import annotations

import math

import numpy as np
import torch

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    FREE_CLASS,
    GROUND_SUPPORT_COUNT,
    OCCUPIED_CLASS,
    OUTPUT_SHAPE,
    PIXEL_RAY_SHAPE,
    SOURCE_SHAPE,
    calibrated_pixel_ray_directions_body_v4,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    DEPTH_BIN_COUNT,
    DEPTH_BIN_SIZE_M,
    DEPTH_FAR_EDGE_M,
    ObservableCameraRayEvidenceV4RawOutput,
    ordered_obstacle_first_hit_log_probabilities_v4,
)
from lewm.models.observable_camera_ray_evidence_v4_training import (
    SoftObservableCameraRayRasterV4,
    balanced_ground_clear_bce_v4,
    calibrated_pixel_ray_directions_torch_v4,
    derive_observable_camera_ray_evidence_v4_targets,
    hierarchical_raster_cross_entropy_v4,
    in_range_pixel_offset_smooth_l1_v4,
    ordered_obstacle_first_hit_nll_breakdown_v4,
    ordered_obstacle_first_hit_nll_v4,
    soft_rasterize_observable_camera_ray_evidence_v4,
)


def _calibration(
    batch: int = 1,
    *,
    origin: tuple[float, float, float] = (-1.05, -3.15, 0.0),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera_origin = torch.tensor(origin, dtype=dtype)[None].expand(
        batch, -1
    ).clone()
    basis = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=dtype,
    )[None].expand(batch, -1, -1).clone()
    return camera_origin, basis


def _ground_labels(
    shape: tuple[int, int] = (2, 3),
) -> tuple[torch.Tensor, torch.Tensor]:
    in_frustum = torch.ones(
        1,
        *shape,
        GROUND_SUPPORT_COUNT,
        dtype=torch.bool,
    )
    clear = in_frustum.clone()
    clear[:, 0, 0, 0] = False
    return in_frustum, clear


def _targets_for_pixel_ranges(
    hit_mask: torch.Tensor,
    hit_distance_m: torch.Tensor,
) -> object:
    in_frustum, clear = _ground_labels()
    return derive_observable_camera_ray_evidence_v4_targets(
        pixel_hit_mask=hit_mask,
        pixel_first_hit_distance_m=hit_distance_m,
        ground_support_in_frustum=in_frustum,
        ground_support_clear_to_target=clear,
    )


def _raw_output(
    hazard_logits: torch.Tensor,
    offset_m: torch.Tensor,
    ground_logits: torch.Tensor,
    *,
    ground_in_frustum: torch.Tensor | None = None,
) -> ObservableCameraRayEvidenceV4RawOutput:
    in_frustum = (
        torch.ones_like(ground_logits, dtype=torch.bool)
        if ground_in_frustum is None
        else ground_in_frustum
    )
    return ObservableCameraRayEvidenceV4RawOutput(
        pixel_first_hit_hazard_logits=hazard_logits,
        pixel_within_bin_offset_m=offset_m,
        ground_clear_to_target_logits=ground_logits,
        ground_query_in_frustum=in_frustum,
        ground_query_uv_px=torch.zeros(
            *ground_logits.shape,
            2,
            dtype=torch.float64,
            device=ground_logits.device,
        ),
        ground_target_distance_m=torch.ones(
            ground_logits.shape,
            dtype=torch.float64,
            device=ground_logits.device,
        ),
    )


def test_target_derivation_bins_offsets_and_censors_far_hits_as_no_hit() -> None:
    hit_mask = torch.tensor(
        [[[True, True, True], [True, True, False]]],
        dtype=torch.bool,
    )
    hit_distance = torch.tensor(
        [[[0.05, 0.149, 0.15], [6.449, DEPTH_FAR_EDGE_M, 0.0]]],
        dtype=torch.float64,
    )

    targets = _targets_for_pixel_ranges(hit_mask, hit_distance)

    torch.testing.assert_close(
        targets.pixel_hit_bin_index,
        torch.tensor([[[0, 0, 1], [63, 0, 0]]]),
    )
    torch.testing.assert_close(
        targets.pixel_in_range_hit_mask,
        torch.tensor(
            [[[True, True, True], [True, False, False]]],
            dtype=torch.bool,
        ),
    )
    torch.testing.assert_close(
        targets.pixel_within_bin_offset_m,
        torch.tensor(
            [[[-0.05, 0.049, -0.05], [0.049, 0.0, 0.0]]],
            dtype=torch.float64,
        ),
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.equal(
        targets.pixel_no_hit_mask,
        ~targets.pixel_in_range_hit_mask,
    )


def test_ordered_nll_includes_hit_and_no_hit_oracles() -> None:
    hit_mask = torch.tensor([[[True, False]]], dtype=torch.bool)
    hit_distance = torch.tensor([[[0.30, 0.0]]])
    targets = _targets_for_pixel_ranges(hit_mask, hit_distance)
    hazard = torch.full(
        (1, DEPTH_BIN_COUNT, 1, 2),
        -20.0,
        requires_grad=True,
    )
    with torch.no_grad():
        hazard[:, 2, 0, 0] = 20.0

    loss = ordered_obstacle_first_hit_nll_v4(hazard, targets)

    assert float(loss.detach()) < 2e-6
    loss.backward()
    assert hazard.grad is not None and torch.isfinite(hazard.grad).all()


def test_ordered_nll_balances_no_hit_and_each_nonempty_hit_distance_bin() -> None:
    torch.manual_seed(19)
    hit_mask = torch.zeros(1, 1, 102, dtype=torch.bool)
    hit_mask[0, 0, 100:] = True
    hit_distance = torch.zeros(1, 1, 102, dtype=torch.float64)
    hit_distance[0, 0, 100] = 0.10
    hit_distance[0, 0, 101] = 1.10
    targets = _targets_for_pixel_ranges(hit_mask, hit_distance)
    hazard = torch.randn(
        1,
        DEPTH_BIN_COUNT,
        1,
        102,
        dtype=torch.float64,
        requires_grad=True,
    )

    breakdown = ordered_obstacle_first_hit_nll_breakdown_v4(hazard, targets)
    ordered = ordered_obstacle_first_hit_log_probabilities_v4(hazard)
    gathered = ordered.hit.gather(
        1,
        targets.pixel_hit_bin_index[:, None],
    ).squeeze(1)
    no_hit_loss = -ordered.no_hit[targets.pixel_no_hit_mask].mean()
    bin_zero_loss = -gathered[
        targets.pixel_in_range_hit_mask
        & (targets.pixel_hit_bin_index == 0)
    ].mean()
    bin_ten_loss = -gathered[
        targets.pixel_in_range_hit_mask
        & (targets.pixel_hit_bin_index == 10)
    ].mean()
    expected = torch.stack((no_hit_loss, bin_zero_loss, bin_ten_loss)).mean()

    torch.testing.assert_close(breakdown.total, expected)
    assert breakdown.no_hit_count == 100
    assert breakdown.hit_distance_bin_counts[0] == 1
    assert breakdown.hit_distance_bin_counts[10] == 1
    assert sum(breakdown.hit_distance_bin_counts) == 2
    assert breakdown.nonempty_group_count == 3
    naive = -torch.where(
        targets.pixel_in_range_hit_mask,
        gathered,
        ordered.no_hit,
    ).mean()
    assert not torch.allclose(breakdown.total, naive)
    breakdown.total.backward()
    assert hazard.grad is not None and torch.isfinite(hazard.grad).all()


def test_in_range_offset_smooth_l1_uses_only_the_target_hit_bin() -> None:
    hit_mask = torch.tensor([[[True, False]]], dtype=torch.bool)
    hit_distance = torch.tensor([[[0.30, 0.0]]])
    targets = _targets_for_pixel_ranges(hit_mask, hit_distance)
    predicted = torch.full(
        (1, DEPTH_BIN_COUNT, 1, 2),
        0.04,
        requires_grad=True,
    )
    with torch.no_grad():
        predicted[:, 2, 0, 0] = targets.pixel_within_bin_offset_m[0, 0, 0]

    loss = in_range_pixel_offset_smooth_l1_v4(predicted, targets)

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    assert predicted.grad is not None
    nonzero = torch.count_nonzero(predicted.grad)
    assert int(nonzero) == 0


def test_balanced_ground_bce_handles_clear_blocked_distance_groups() -> None:
    in_frustum = torch.ones(1, 2, 2, 5, dtype=torch.bool)
    clear = torch.zeros_like(in_frustum)
    clear[:, 0] = True
    clear[:, 1, 1, :3] = True
    targets = derive_observable_camera_ray_evidence_v4_targets(
        pixel_hit_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        pixel_first_hit_distance_m=torch.zeros(1, 1, 1),
        ground_support_in_frustum=in_frustum,
        ground_support_clear_to_target=clear,
    )
    distance = torch.empty(1, 2, 2, 5)
    distance[:, 0] = 0.5
    distance[:, 1] = 2.5
    logits = torch.where(clear, torch.tensor(20.0), torch.tensor(-20.0))
    logits.requires_grad_(True)

    loss = balanced_ground_clear_bce_v4(logits, targets, distance)

    assert float(loss.detach()) < 3e-9
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_torch_calibrated_full_pixel_rays_match_frozen_contract() -> None:
    _origin, basis = _calibration(dtype=torch.float64)

    actual = calibrated_pixel_ray_directions_torch_v4(
        basis,
        ray_shape=PIXEL_RAY_SHAPE,
        dtype=torch.float64,
    )
    expected = calibrated_pixel_ray_directions_body_v4(basis[0].numpy())

    np.testing.assert_allclose(
        actual[0].numpy(),
        expected,
        rtol=0.0,
        atol=2e-15,
    )


def test_soft_raster_oracle_is_free_without_hits_and_occupied_takes_precedence() -> None:
    ground_logits = torch.full((1, 4, 4, 5), 20.0)
    no_hit_hazard = torch.full((1, 64, 1, 1), -20.0)
    offset = torch.zeros_like(no_hit_hazard)
    origin, basis = _calibration()

    free_raster = soft_rasterize_observable_camera_ray_evidence_v4(
        _raw_output(no_hit_hazard, offset, ground_logits),
        camera_origin_body_m=origin,
        camera_basis_body_fru=basis,
        output_shape=(2, 2),
        pixel_ray_chunk_size=1,
    )
    assert torch.all(free_raster.class_probabilities[:, FREE_CLASS] > 0.99999)

    hit_hazard = no_hit_hazard.clone()
    hit_hazard[:, 0] = 20.0
    occupied_raster = soft_rasterize_observable_camera_ray_evidence_v4(
        _raw_output(hit_hazard, offset, ground_logits),
        camera_origin_body_m=origin,
        camera_basis_body_fru=basis,
        output_shape=(2, 2),
        pixel_ray_chunk_size=1,
    )
    assert float(occupied_raster.occupied_probability[0, 0, 0]) > 0.99999
    assert (
        float(occupied_raster.class_probabilities[0, OCCUPIED_CLASS, 0, 0])
        > 0.99999
    )
    assert float(occupied_raster.class_probabilities[0, FREE_CLASS, 0, 0]) < 1e-5

    target = torch.full((1, 2, 2), FREE_CLASS, dtype=torch.uint8)
    target[0, 0, 0] = OCCUPIED_CLASS
    hierarchical = hierarchical_raster_cross_entropy_v4(
        occupied_raster,
        target,
    )
    assert float(hierarchical.total) < 1e-4


def test_soft_raster_and_hierarchical_loss_have_finite_branch_gradients() -> None:
    torch.manual_seed(17)
    hazard = torch.full((1, 64, 1, 1), -20.0)
    hazard[:, 0] = -1.0
    hazard.requires_grad_(True)
    offset = torch.zeros(1, 64, 1, 1, requires_grad=True)
    ground_logits = torch.randn(1, 4, 4, 5, requires_grad=True)
    origin, basis = _calibration(origin=(-1.0, -3.15, 0.0))
    raster = soft_rasterize_observable_camera_ray_evidence_v4(
        _raw_output(hazard, offset, ground_logits),
        camera_origin_body_m=origin,
        camera_basis_body_fru=basis,
        output_shape=(2, 2),
        pixel_ray_chunk_size=1,
    )
    target = torch.tensor(
        [[[OCCUPIED_CLASS, FREE_CLASS], [0, FREE_CLASS]]],
        dtype=torch.uint8,
    )

    loss = hierarchical_raster_cross_entropy_v4(raster, target).total
    loss.backward()

    for value in (hazard, offset, ground_logits):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
    assert bool((offset.grad.abs() > 0.0).any().item())


def test_hierarchical_raster_loss_balances_rare_states_and_reports_counts() -> None:
    occupied = torch.full((1, 1, 101), 0.10, requires_grad=True)
    conditional_free = torch.full((1, 1, 101), 0.10, requires_grad=True)
    not_occupied = 1.0 - occupied
    classes = torch.stack(
        (
            not_occupied * (1.0 - conditional_free),
            not_occupied * conditional_free,
            occupied,
        ),
        dim=1,
    )
    raster = SoftObservableCameraRayRasterV4(
        source_free_probability=torch.ones(1, 2, 2),
        free_given_not_occupied_probability=conditional_free,
        occupied_probability=occupied,
        class_probabilities=classes,
    )
    target = torch.zeros(1, 1, 101, dtype=torch.uint8)
    target[0, 0, 99] = FREE_CLASS
    target[0, 0, 100] = OCCUPIED_CLASS

    loss = hierarchical_raster_cross_entropy_v4(raster, target)

    rare_positive = -math.log(0.10)
    common_negative = -math.log(0.90)
    expected_component = 0.5 * (rare_positive + common_negative)
    torch.testing.assert_close(
        loss.occupied,
        loss.occupied.new_tensor(expected_component),
    )
    torch.testing.assert_close(
        loss.free_given_not_occupied,
        loss.free_given_not_occupied.new_tensor(expected_component),
    )
    torch.testing.assert_close(
        loss.total,
        loss.total.new_tensor(expected_component),
    )
    assert loss.occupied_count == 1
    assert loss.rest_count == 100
    assert loss.free_count == 1
    assert loss.unknown_count == 99
    loss.total.backward()
    assert occupied.grad is not None and torch.isfinite(occupied.grad).all()
    assert conditional_free.grad is not None
    assert torch.isfinite(conditional_free.grad).all()


def test_hierarchical_empty_conditional_group_keeps_zero_gradient_path() -> None:
    occupied = torch.full((1, 2, 3), 0.7, requires_grad=True)
    conditional_free = torch.full((1, 2, 3), 0.4, requires_grad=True)
    classes = torch.stack(
        (
            (1.0 - occupied) * (1.0 - conditional_free),
            (1.0 - occupied) * conditional_free,
            occupied,
        ),
        dim=1,
    )
    raster = SoftObservableCameraRayRasterV4(
        source_free_probability=torch.ones(1, 4, 6),
        free_given_not_occupied_probability=conditional_free,
        occupied_probability=occupied,
        class_probabilities=classes,
    )
    target = torch.full((1, 2, 3), OCCUPIED_CLASS, dtype=torch.uint8)

    loss = hierarchical_raster_cross_entropy_v4(raster, target)
    gradient = torch.autograd.grad(
        loss.total,
        (occupied, conditional_free),
    )

    assert loss.rest_count == loss.free_count == loss.unknown_count == 0
    assert loss.occupied_count == 6
    torch.testing.assert_close(
        loss.free_given_not_occupied,
        torch.zeros_like(loss.free_given_not_occupied),
    )
    assert torch.isfinite(gradient[0]).all()
    torch.testing.assert_close(gradient[1], torch.zeros_like(gradient[1]))


def test_ground_conjunctions_match_log_space_reference() -> None:
    hazard = torch.full((1, 64, 1, 1), -20.0, dtype=torch.float64)
    offset = torch.zeros_like(hazard)
    ground_logits = torch.full((1, 4, 4, 5), -3.0, dtype=torch.float64)
    origin, basis = _calibration(dtype=torch.float64)

    raster = soft_rasterize_observable_camera_ray_evidence_v4(
        _raw_output(hazard, offset, ground_logits),
        camera_origin_body_m=origin,
        camera_basis_body_fru=basis,
        output_shape=(2, 2),
        pixel_ray_chunk_size=1,
    )

    log_clear = torch.nn.functional.logsigmoid(torch.tensor(-3.0, dtype=torch.float64))
    torch.testing.assert_close(
        raster.source_free_probability,
        torch.ones_like(raster.source_free_probability) * (5.0 * log_clear).exp(),
        atol=1e-18,
        rtol=1e-14,
    )
    torch.testing.assert_close(
        raster.free_given_not_occupied_probability,
        torch.ones_like(raster.free_given_not_occupied_probability)
        * (20.0 * log_clear).exp(),
        atol=1e-24,
        rtol=1e-14,
    )


def test_full_contract_soft_raster_is_finite_and_normalized() -> None:
    hazard = torch.full(
        (1, DEPTH_BIN_COUNT, *PIXEL_RAY_SHAPE),
        -12.0,
    )
    offset = torch.zeros_like(hazard)
    ground_logits = torch.zeros(
        1,
        *SOURCE_SHAPE,
        GROUND_SUPPORT_COUNT,
    )
    origin, basis = _calibration(origin=(0.326, 0.0, 0.043))

    raster = soft_rasterize_observable_camera_ray_evidence_v4(
        _raw_output(hazard, offset, ground_logits),
        camera_origin_body_m=origin,
        camera_basis_body_fru=basis,
        output_shape=OUTPUT_SHAPE,
        pixel_ray_chunk_size=512,
    )

    assert raster.source_free_probability.shape == (1, *SOURCE_SHAPE)
    assert raster.class_probabilities.shape == (1, 3, *OUTPUT_SHAPE)
    assert torch.isfinite(raster.class_probabilities).all()
    torch.testing.assert_close(
        raster.class_probabilities.sum(dim=1),
        torch.ones(1, *OUTPUT_SHAPE),
        atol=2e-6,
        rtol=2e-6,
    )
