from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from lewm.models.observable_camera_ray_evidence_v4 import (
    DEPTH_FAR_EDGE_M,
    DEPTH_NEAR_EDGE_M,
    GroundQueryGeometryV4,
    ObservableCameraRayEvidenceV4Model,
    ordered_obstacle_first_hit_log_probabilities_v4,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v14_unified_ray_survival import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV14,
    MIGRATED_EVIDENCE_STATE_COUNT_V14,
    ONLINE_TRAINABLE_PARAMETER_COUNT_V13,
    ONLINE_TRAINABLE_PARAMETER_COUNT_V14,
    PREDICTOR_GROUP_PARAMETER_COUNT_V14,
    PROJECTION_INITIALIZATION_SEED_V13,
    PROJECTION_INITIALIZATION_SEED_V14,
    REPRESENTATION_GROUP_PARAMETER_COUNT_V14,
    SHARED_ROUTE_PARAMETER_COUNT_V13,
    SHARED_ROUTE_PARAMETER_COUNT_V14,
    TARGET_BOTTLENECK_PARAMETER_COUNT_V13,
    TARGET_BOTTLENECK_PARAMETER_COUNT_V14,
    finite_ground_clear_logits_v14,
    fractional_ray_log_survival_v14,
    sample_hazards_at_ground_queries_v14,
)


def _query(
    distances_m: torch.Tensor,
    *,
    sample_grid: torch.Tensor | None = None,
    in_frustum: torch.Tensor | None = None,
) -> GroundQueryGeometryV4:
    distances_m = distances_m.reshape(1, 1, 1, -1).to(dtype=torch.float64)
    query_shape = tuple(distances_m.shape)
    if sample_grid is None:
        sample_grid = torch.zeros((*query_shape, 2), dtype=torch.float64)
    else:
        sample_grid = sample_grid.reshape(*query_shape, 2).to(dtype=torch.float64)
    if in_frustum is None:
        in_frustum = torch.ones(query_shape, dtype=torch.bool)
    return GroundQueryGeometryV4(
        in_frustum=in_frustum,
        uv_px=torch.zeros((*query_shape, 2), dtype=torch.float64),
        target_distance_m=distances_m,
        sample_grid=sample_grid,
    )


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def fit_model() -> ObservableCameraRayEvidenceV4Model:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(14_001)
        return ObservableCameraRayEvidenceV4Model().eval()
    finally:
        torch.random.set_rng_state(caller_rng)


@pytest.fixture(scope="module")
def v14_model(
    fit_model: ObservableCameraRayEvidenceV4Model,
) -> GeometryAnchoredSweptProgressSurvivalJointJepaV14:
    caller_rng = torch.random.get_rng_state().clone()
    model = GeometryAnchoredSweptProgressSurvivalJointJepaV14(
        fit_model,
        _sweep_masks(),
    ).eval()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    return model


def test_fractional_survival_closed_form_and_finite_logit_boundary() -> None:
    distances = torch.tensor(
        (
            DEPTH_NEAR_EDGE_M,
            DEPTH_NEAR_EDGE_M + 0.05,
            DEPTH_NEAR_EDGE_M + 0.10,
            DEPTH_NEAR_EDGE_M + 0.15,
            DEPTH_FAR_EDGE_M,
            DEPTH_FAR_EDGE_M + 1.0,
        ),
        dtype=torch.float64,
    )
    hazard = torch.zeros((1, 64, 1, 1), dtype=torch.float64)
    log_survival = fractional_ray_log_survival_v14(
        hazard,
        _query(distances),
        query_chunk_size=2,
    ).reshape(-1)
    expected = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 64.0, 64.0),
        dtype=torch.float64,
    ) * math.log(0.5)
    torch.testing.assert_close(log_survival, expected, rtol=0.0, atol=1e-12)
    assert log_survival[0].item() == 0.0

    logits = finite_ground_clear_logits_v14(log_survival)
    assert bool(torch.isfinite(logits).all())
    assert logits[0] > logits[1]
    torch.testing.assert_close(
        torch.sigmoid(logits[1:]),
        log_survival[1:].exp(),
        rtol=1e-12,
        atol=0.0,
    )


def test_survival_is_monotone_and_far_edge_is_existing_no_hit_identity() -> None:
    per_bin = torch.linspace(-1.0, 1.0, 64, dtype=torch.float64)
    hazard = per_bin.reshape(1, 64, 1, 1).expand(1, 64, 2, 3).contiguous()
    distances = torch.linspace(
        DEPTH_NEAR_EDGE_M,
        DEPTH_FAR_EDGE_M,
        17,
        dtype=torch.float64,
    )
    query = _query(distances)
    unchunked = fractional_ray_log_survival_v14(
        hazard,
        query,
        query_chunk_size=None,
    ).reshape(-1)
    chunked = fractional_ray_log_survival_v14(
        hazard,
        query,
        query_chunk_size=3,
    ).reshape(-1)
    assert torch.equal(unchunked, chunked)
    assert bool((unchunked[1:] <= unchunked[:-1]).all())

    existing_no_hit = ordered_obstacle_first_hit_log_probabilities_v4(
        per_bin.reshape(1, 64, 1, 1)
    ).no_hit.squeeze()
    torch.testing.assert_close(
        unchunked[-1],
        existing_no_hit,
        rtol=0.0,
        atol=1e-12,
    )


def test_boundary_sampling_uses_border_padding_with_align_corners_false() -> None:
    spatial = torch.tensor(((1.0, 2.0), (3.0, 4.0)), dtype=torch.float32)
    hazard = spatial[None, None].expand(1, 64, 2, 2).contiguous()
    grids = torch.tensor(((-1.0, -1.0), (1.0, 1.0), (1.0, 0.0)))
    query = _query(torch.full((3,), 1.0), sample_grid=grids)
    sampled = sample_hazards_at_ground_queries_v14(
        hazard,
        query,
        query_chunk_size=1,
    )
    assert tuple(sampled.shape) == (1, 1, 1, 3, 64)
    torch.testing.assert_close(
        sampled[0, 0, 0, :, 0],
        torch.tensor((1.0, 4.0, 3.0)),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        sampled,
        sample_hazards_at_ground_queries_v14(
            hazard,
            query,
            query_chunk_size=None,
        ),
    )


def test_selected_migration_exact_counts_and_shared_ground_gradients(
    fit_model: ObservableCameraRayEvidenceV4Model,
    v14_model: GeometryAnchoredSweptProgressSurvivalJointJepaV14,
) -> None:
    head = v14_model.bev_lift.evidence_head
    assert not hasattr(head, "ground_head")
    assert len(v14_model.bev_lift.migrated_evidence_state_names) == (
        MIGRATED_EVIDENCE_STATE_COUNT_V14
    )
    assert set(v14_model.bev_lift.migrated_evidence_state_names) == set(
        head.state_dict()
    )
    fit_state = fit_model.state_dict()
    assert all(
        torch.equal(value, fit_state[name])
        for name, value in head.state_dict().items()
    )

    groups = v14_model.trainable_parameter_groups_v14()
    assert tuple(
        sum(parameter.numel() for _, parameter in group) for group in groups
    ) == (
        SHARED_ROUTE_PARAMETER_COUNT_V14,
        REPRESENTATION_GROUP_PARAMETER_COUNT_V14,
        PREDICTOR_GROUP_PARAMETER_COUNT_V14,
    )
    assert sum(
        parameter.numel()
        for parameter in v14_model.parameters()
        if parameter.requires_grad
    ) == ONLINE_TRAINABLE_PARAMETER_COUNT_V14
    assert sum(
        parameter.numel()
        for module in v14_model.target_modules()
        for parameter in module.parameters()
    ) == TARGET_BOTTLENECK_PARAMETER_COUNT_V14
    assert SHARED_ROUTE_PARAMETER_COUNT_V13 == SHARED_ROUTE_PARAMETER_COUNT_V14
    assert ONLINE_TRAINABLE_PARAMETER_COUNT_V13 == (
        ONLINE_TRAINABLE_PARAMETER_COUNT_V14
    )
    assert TARGET_BOTTLENECK_PARAMETER_COUNT_V13 == (
        TARGET_BOTTLENECK_PARAMETER_COUNT_V14
    )
    assert (
        PROJECTION_INITIALIZATION_SEED_V13
        == PROJECTION_INITIALIZATION_SEED_V14
        == 20_260_729
    )

    dense = torch.randn(
        (1, 36, 112, 112),
        generator=torch.Generator().manual_seed(140),
        requires_grad=True,
    )
    hazard, _ = head.pixel_branch(dense)
    query = _query(torch.tensor((1.05,)))
    logits = head.ground_survival_branch(
        hazard,
        query,
        query_chunk_size=1,
    )
    F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
    assert dense.grad is not None and bool(torch.isfinite(dense.grad).all())
    assert head.pixel_head.weight.grad is not None
    assert bool(torch.isfinite(head.pixel_head.weight.grad).all())
    assert int(torch.count_nonzero(head.pixel_head.weight.grad)) > 0


def test_v14_lift_preserves_v13_state_shapes(
    v14_model: GeometryAnchoredSweptProgressSurvivalJointJepaV14,
) -> None:
    tokens = torch.randn(
        (1, 256, 192),
        generator=torch.Generator().manual_seed(141),
    )
    with torch.no_grad():
        encoded = v14_model.bev_lift.forward_with_evidence(tokens)
        auxiliary = v14_model.bev_lift.forward_with_auxiliary_evidence(
            tokens,
            camera_origin_body_m=v14_model.bev_lift.nominal_camera_origin_body_m[
                None
            ],
            camera_basis_body_fru=v14_model.bev_lift.nominal_camera_basis_body_fru[
                None
            ],
            ground_plane_z_body_m=v14_model.bev_lift.nominal_ground_plane_z_body_m[
                None
            ],
        )
    assert tuple(encoded.latent.shape) == (1, 64, 64, 64)
    assert tuple(encoded.free_evidence_planes.shape) == (1, 40, 64, 64)
    assert tuple(encoded.occupied_evidence_planes.shape) == (1, 64, 64, 64)
    assert tuple(
        encoded.nominal_evidence.pixel_first_hit_hazard_logits.shape
    ) == (1, 64, 84, 112)
    assert tuple(encoded.nominal_evidence.ground_clear_to_target_logits.shape) == (
        1,
        128,
        128,
        5,
    )
    assert bool(torch.isfinite(encoded.latent).all())
    assert bool(
        torch.isfinite(
            encoded.nominal_evidence.ground_clear_to_target_logits
        ).all()
    )
    assert torch.equal(encoded.latent, auxiliary.latent)
    assert torch.equal(
        auxiliary.nominal_evidence.ground_clear_to_target_logits,
        auxiliary.auxiliary_evidence.ground_clear_to_target_logits,
    )
    assert auxiliary.nominal_evidence.pixel_first_hit_hazard_logits is (
        auxiliary.auxiliary_evidence.pixel_first_hit_hazard_logits
    )
