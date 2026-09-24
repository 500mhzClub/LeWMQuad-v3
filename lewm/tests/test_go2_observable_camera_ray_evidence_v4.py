from __future__ import annotations

import numpy as np
import pytest

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
    CAMERA_NEAR_M,
    EVIDENCE_SCHEMA,
    FREE_CLASS,
    GROUND_SUPPORT_COUNT,
    OCCUPIED_CLASS,
    OUTPUT_SHAPE,
    PIXEL_RAY_SHAPE,
    SOURCE_SHAPE,
    UNKNOWN_CLASS,
    ObservableCameraRayEvidenceV4,
    calibrated_pixel_ray_directions_body_v4,
    canonical_ground_support_points_body_m,
    observable_camera_ray_evidence_v4_from_mapping,
    project_body_points_to_camera_v4,
    project_canonical_ground_support_v4,
    rasterize_observable_camera_ray_evidence_v4,
)


DOWNWARD_BASIS_FRU = np.asarray(
    ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    dtype=np.float32,
)
DEFAULT_ORIGIN = np.asarray((2.2, 0.0, 10.0), dtype=np.float32)
GROUND_Z = 0.0


def _evidence(
    *,
    all_ground_clear: bool = False,
    hit_pixel_distances: tuple[tuple[int, int, float], ...] = (),
    camera_origin: np.ndarray = DEFAULT_ORIGIN,
    camera_basis: np.ndarray = DOWNWARD_BASIS_FRU,
) -> ObservableCameraRayEvidenceV4:
    projection = project_canonical_ground_support_v4(
        camera_origin_body_m=camera_origin,
        camera_basis_body_fru=camera_basis,
        ground_plane_z_body_m=GROUND_Z,
    )
    clear = projection.in_frustum.copy() if all_ground_clear else np.zeros_like(
        projection.in_frustum
    )
    hit_mask = np.zeros(PIXEL_RAY_SHAPE, dtype=bool)
    hit_distance = np.zeros(PIXEL_RAY_SHAPE, dtype=np.float32)
    for row, column, distance in hit_pixel_distances:
        hit_mask[row, column] = True
        hit_distance[row, column] = distance
    return ObservableCameraRayEvidenceV4(
        camera_origin_body_m=camera_origin,
        camera_basis_body_fru=camera_basis,
        ground_plane_z_body_m=GROUND_Z,
        ground_support_in_frustum=projection.in_frustum,
        ground_support_clear_to_target=clear,
        pixel_hit_mask=hit_mask,
        pixel_first_hit_distance_m=hit_distance,
    )


def _mapping() -> dict[str, object]:
    evidence = _evidence()
    return {
        "schema": EVIDENCE_SCHEMA,
        "camera_origin_body_m": evidence.camera_origin_body_m.copy(),
        "camera_basis_body_fru": evidence.camera_basis_body_fru.copy(),
        "ground_plane_z_body_m": evidence.ground_plane_z_body_m,
        "ground_support_in_frustum": evidence.ground_support_in_frustum.copy(),
        "ground_support_clear_to_target": (
            evidence.ground_support_clear_to_target.copy()
        ),
        "pixel_hit_mask": evidence.pixel_hit_mask.copy(),
        "pixel_first_hit_distance_m": (
            evidence.pixel_first_hit_distance_m.copy()
        ),
    }


def _evidence_with_hit_xy(hit_xy: tuple[float, float]) -> ObservableCameraRayEvidenceV4:
    row, column = 40, 55
    distance = 10.0
    direction = calibrated_pixel_ray_directions_body_v4(DOWNWARD_BASIS_FRU)[
        row, column
    ]
    origin = np.asarray(
        (
            hit_xy[0] - direction[0] * distance,
            hit_xy[1] - direction[1] * distance,
            10.0,
        ),
        dtype=np.float32,
    )
    return _evidence(
        camera_origin=origin,
        hit_pixel_distances=((row, column, distance),),
    )


def test_free_requires_all_five_supports_in_all_four_aligned_source_cells() -> None:
    evidence = _evidence(all_ground_clear=True)
    clear = evidence.ground_support_clear_to_target.copy()
    clear[0, 0, 0] = False
    evidence = ObservableCameraRayEvidenceV4(
        camera_origin_body_m=evidence.camera_origin_body_m,
        camera_basis_body_fru=evidence.camera_basis_body_fru,
        ground_plane_z_body_m=evidence.ground_plane_z_body_m,
        ground_support_in_frustum=evidence.ground_support_in_frustum,
        ground_support_clear_to_target=clear,
        pixel_hit_mask=evidence.pixel_hit_mask,
        pixel_first_hit_distance_m=evidence.pixel_first_hit_distance_m,
    )

    raster = rasterize_observable_camera_ray_evidence_v4(evidence)

    assert not raster.source_free_mask[0, 0]
    assert raster.source_free_mask[0, 1]
    assert raster.output_labels[0, 0] == UNKNOWN_CLASS
    assert raster.output_labels[0, 1] == FREE_CLASS
    assert raster.output_labels[1, 0] == FREE_CLASS
    assert np.count_nonzero(raster.output_labels == UNKNOWN_CLASS) == 1


def test_pixel_hit_on_closed_internal_boundary_marks_all_four_cells() -> None:
    raster = rasterize_observable_camera_ray_evidence_v4(
        _evidence_with_hit_xy((-0.9, -3.1))
    )

    expected = np.zeros(OUTPUT_SHAPE, dtype=bool)
    expected[0:2, 0:2] = True
    np.testing.assert_array_equal(raster.output_occupied_mask, expected)


@pytest.mark.parametrize(
    ("hit_xy", "expected_cell"),
    [((-1.0, -3.2), (0, 0)), ((5.4, 3.2), (63, 63))],
)
def test_closed_outer_boundary_is_included_once(
    hit_xy: tuple[float, float], expected_cell: tuple[int, int]
) -> None:
    raster = rasterize_observable_camera_ray_evidence_v4(
        _evidence_with_hit_xy(hit_xy)
    )
    assert raster.output_occupied_mask[expected_cell]
    assert np.count_nonzero(raster.output_occupied_mask) == 1


def test_occupied_has_precedence_over_free() -> None:
    row, column = 41, 56
    direction = calibrated_pixel_ray_directions_body_v4(DOWNWARD_BASIS_FRU)[
        row, column
    ]
    distance = float((GROUND_Z - DEFAULT_ORIGIN[2]) / direction[2])
    evidence = _evidence(
        all_ground_clear=True,
        hit_pixel_distances=((row, column, distance),),
    )
    raster = rasterize_observable_camera_ray_evidence_v4(evidence)
    occupied = np.argwhere(raster.output_occupied_mask)
    assert occupied.size
    r, c = occupied[0]
    assert raster.output_free_before_occupied_mask[r, c]
    assert raster.output_labels[r, c] == OCCUPIED_CLASS


def test_distances_are_authoritative_and_xy_is_deterministically_derived() -> None:
    row, column = 20, 30
    distance = 2.5
    evidence = _evidence(hit_pixel_distances=((row, column, distance),))
    direction = calibrated_pixel_ray_directions_body_v4(DOWNWARD_BASIS_FRU)[
        row, column
    ]
    expected = DEFAULT_ORIGIN.astype(np.float64)[:2] + direction[:2] * distance
    np.testing.assert_allclose(evidence.pixel_hit_xy_body_m[row, column], expected)
    assert np.count_nonzero(evidence.pixel_hit_xy_body_m) == 2


def test_rasterization_and_hashes_are_deterministic_and_immutable() -> None:
    first_evidence = _evidence(
        all_ground_clear=True,
        hit_pixel_distances=((20, 30, 2.5), (40, 55, 10.0)),
    )
    second_evidence = _evidence(
        all_ground_clear=True,
        hit_pixel_distances=((20, 30, 2.5), (40, 55, 10.0)),
    )
    first = rasterize_observable_camera_ray_evidence_v4(first_evidence)
    second = rasterize_observable_camera_ray_evidence_v4(second_evidence)

    assert first_evidence.content_sha256() == second_evidence.content_sha256()
    assert first.content_sha256() == second.content_sha256()
    np.testing.assert_array_equal(first.output_labels, second.output_labels)
    with pytest.raises(ValueError):
        first_evidence.pixel_first_hit_distance_m[0, 0] = 1.0
    with pytest.raises(ValueError):
        first.output_labels[0, 0] = UNKNOWN_CLASS


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "physical_free_mask",
        "collision_boxes",
        "collision_overlap",
        "morphology",
        "body_inflation_radius_m",
        "configuration_occupancy",
    ],
)
def test_external_mapping_rejects_forbidden_privileged_fields(
    forbidden_field: str,
) -> None:
    payload = _mapping()
    payload[forbidden_field] = np.zeros((1,), dtype=bool)
    with pytest.raises(ValueError, match="forbids privileged fields"):
        observable_camera_ray_evidence_v4_from_mapping(payload)


def test_evidence_rejects_calibration_mask_disagreement() -> None:
    payload = _mapping()
    mask = payload["ground_support_in_frustum"]
    assert isinstance(mask, np.ndarray)
    mask[0, 0, 0] = not bool(mask[0, 0, 0])
    with pytest.raises(ValueError, match="disagrees with frame calibration"):
        observable_camera_ray_evidence_v4_from_mapping(payload)


def test_evidence_rejects_clear_support_outside_camera_frustum() -> None:
    payload = _mapping()
    origin = np.asarray((0.326, 0.0, 0.043), dtype=np.float32)
    basis = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    projection = project_canonical_ground_support_v4(
        camera_origin_body_m=origin,
        camera_basis_body_fru=basis,
        ground_plane_z_body_m=-0.35,
    )
    payload["camera_origin_body_m"] = origin
    payload["camera_basis_body_fru"] = basis
    payload["ground_plane_z_body_m"] = -0.35
    payload["ground_support_in_frustum"] = projection.in_frustum.copy()
    payload["ground_support_clear_to_target"] = np.zeros_like(
        projection.in_frustum
    )
    clear = payload["ground_support_clear_to_target"]
    mask = payload["ground_support_in_frustum"]
    assert isinstance(clear, np.ndarray) and isinstance(mask, np.ndarray)
    index = tuple(np.argwhere(~mask)[0])
    clear[index] = True
    with pytest.raises(ValueError, match="out-of-frustum"):
        observable_camera_ray_evidence_v4_from_mapping(payload)


def test_evidence_rejects_distance_for_a_pixel_without_a_hit() -> None:
    payload = _mapping()
    distance = payload["pixel_first_hit_distance_m"]
    assert isinstance(distance, np.ndarray)
    distance[0, 0] = 1.0
    with pytest.raises(ValueError, match="canonical zero"):
        observable_camera_ray_evidence_v4_from_mapping(payload)


def test_evidence_rejects_hit_inside_near_plane() -> None:
    payload = _mapping()
    mask = payload["pixel_hit_mask"]
    distance = payload["pixel_first_hit_distance_m"]
    assert isinstance(mask, np.ndarray) and isinstance(distance, np.ndarray)
    mask[0, 0] = True
    distance[0, 0] = CAMERA_NEAR_M / 2.0
    with pytest.raises(ValueError, match="near plane"):
        observable_camera_ray_evidence_v4_from_mapping(payload)


def test_camera_basis_must_be_orthonormal_and_right_handed() -> None:
    payload = _mapping()
    basis = payload["camera_basis_body_fru"]
    assert isinstance(basis, np.ndarray)
    basis[0, 0] = 0.5
    with pytest.raises(ValueError, match="orthonormal"):
        observable_camera_ray_evidence_v4_from_mapping(payload)


def test_arbitrary_ground_query_projection_is_calibrated() -> None:
    points = np.asarray(((2.2, 0.0, 0.0), (20.0, 0.0, 0.0)))
    projected = project_body_points_to_camera_v4(
        points,
        camera_origin_body_m=DEFAULT_ORIGIN,
        camera_basis_body_fru=DOWNWARD_BASIS_FRU,
    )
    assert projected.in_frustum.tolist() == [True, False]
    np.testing.assert_allclose(projected.uv_px[0], (112.0, 84.0))
    np.testing.assert_allclose(projected.target_distance_m[0], 10.0)


def test_canonical_ground_supports_are_cell_center_and_closed_corners() -> None:
    points = canonical_ground_support_points_body_m(ground_z_body_m=-0.31)
    assert points.shape == (*SOURCE_SHAPE, GROUND_SUPPORT_COUNT, 3)
    np.testing.assert_allclose(
        points[0, 0],
        np.asarray(
            [
                (-0.975, -3.175, -0.31),
                (-1.0, -3.2, -0.31),
                (-1.0, -3.15, -0.31),
                (-0.95, -3.2, -0.31),
                (-0.95, -3.15, -0.31),
            ]
        ),
    )
    with pytest.raises(ValueError):
        points[0, 0, 0, 0] = 0.0
