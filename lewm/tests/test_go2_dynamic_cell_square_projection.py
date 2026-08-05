from __future__ import annotations

import math

import pytest

from lewm.benchmarks.go2_dynamic_cell_square_projection import (
    CAMERA_NEAR_M,
    HORIZONTAL_FOV_DEG,
    VERTICAL_FOV_DEG,
    build_dynamic_cell_square_support_mask,
    camera_coordinates_in_frustum,
    cell_center,
    cell_square_query_visible,
    compose_yaw_aligned_camera,
    support_mask_sha256,
)


EXPECTED_LEVEL_SUPPORT_SHA256 = (
    "4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b"
)


def _support_count(mask: tuple[tuple[bool, ...], ...]) -> int:
    return sum(sum(row) for row in mask)


def test_zero_attitude_reproduces_frozen_cell_square_support() -> None:
    first = build_dynamic_cell_square_support_mask((0.0, 0.0, 0.0, 1.0), 0.0)
    second = build_dynamic_cell_square_support_mask((0, 0, 0, 1), 0)

    assert first == second
    assert len(first) == 64
    assert all(len(row) == 64 for row in first)
    assert all(type(value) is bool for row in first for value in row)
    assert _support_count(first) == 2062
    assert support_mask_sha256(first) == EXPECTED_LEVEL_SUPPORT_SHA256
    assert support_mask_sha256(second) == EXPECTED_LEVEL_SUPPORT_SHA256


def test_grid_centers_are_the_exact_registered_body_yaw_grid() -> None:
    assert cell_center(0, 0) == pytest.approx((-0.95, -3.15), abs=1e-15)
    assert cell_center(63, 63) == pytest.approx((5.35, 3.15), abs=1e-15)


def test_full_rectilinear_frustum_boundaries_are_inclusive() -> None:
    horizontal = CAMERA_NEAR_M * math.tan(
        math.radians(HORIZONTAL_FOV_DEG) * 0.5
    )
    vertical = CAMERA_NEAR_M * math.tan(
        math.radians(VERTICAL_FOV_DEG) * 0.5
    )

    for left in (-horizontal, horizontal):
        for up in (-vertical, vertical):
            assert camera_coordinates_in_frustum(CAMERA_NEAR_M, left, up)

    assert not camera_coordinates_in_frustum(
        math.nextafter(CAMERA_NEAR_M, -math.inf), 0.0, 0.0
    )
    assert not camera_coordinates_in_frustum(
        CAMERA_NEAR_M, math.nextafter(horizontal, math.inf), 0.0
    )
    assert not camera_coordinates_in_frustum(
        CAMERA_NEAR_M, 0.0, math.nextafter(vertical, math.inf)
    )


@pytest.mark.parametrize(
    ("quaternion", "yaw", "error"),
    (
        ((0.0, 0.0, 0.0), 0.0, ValueError),
        ((0.0, 0.0, 0.0, 0.0), 0.0, ValueError),
        ((0.0, 0.0, 0.0, 1.1), 0.0, ValueError),
        ((0.0, 0.0, 0.0, float("nan")), 0.0, ValueError),
        ((0.0, 0.0, 0.0, 1.0), float("inf"), ValueError),
        ((0.0, 0.0, math.sin(0.1), math.cos(0.1)), 0.0, ValueError),
        ((False, 0.0, 0.0, 1.0), 0.0, TypeError),
        ((0.0, 0.0, 0.0, 1.0), True, TypeError),
        ((0.0, 0.0, 0.0, 1.0), "0.0", TypeError),
    ),
)
def test_invalid_quaternion_and_yaw_fail_closed(
    quaternion: object,
    yaw: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        compose_yaw_aligned_camera(quaternion, yaw)


@pytest.mark.parametrize(
    "coordinates",
    (
        (True, 0.0, 0.0),
        (0.1, False, 0.0),
        (0.1, 0.0, True),
        ("0.1", 0.0, 0.0),
    ),
)
def test_frustum_coordinates_reject_bool_and_nonnumeric_values(
    coordinates: tuple[object, object, object],
) -> None:
    with pytest.raises(TypeError):
        camera_coordinates_in_frustum(*coordinates)


@pytest.mark.parametrize("row,column", ((True, 0), (0, False), (0.0, 0)))
def test_grid_indices_reject_bool_and_noninteger_values(
    row: object,
    column: object,
) -> None:
    with pytest.raises(TypeError):
        cell_center(row, column)


def test_full_quaternion_composition_preserves_roll_and_pitch() -> None:
    roll = 0.18
    pitch = -0.11
    yaw = 0.37
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    quaternion = (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )

    camera = compose_yaw_aligned_camera(quaternion, yaw)

    assert camera.forward_xyz[2] == pytest.approx(-math.sin(pitch), abs=1e-15)
    assert camera.left_xyz[2] == pytest.approx(
        math.cos(pitch) * math.sin(roll), abs=1e-15
    )
    assert camera.up_xyz[2] == pytest.approx(
        math.cos(pitch) * math.cos(roll), abs=1e-15
    )
    assert camera.forward_xyz != (1.0, 0.0, 0.0)
    assert camera.up_xyz != (0.0, 0.0, 1.0)


def test_nontrivial_pose_matches_frozen_analytic_basis_and_mount_origin() -> None:
    """Freeze Rz(-yaw) Rz(yaw) Ry(pitch) Rx(roll) independently."""

    quaternion = (
        0.09828232983566462,
        -0.037307702965734074,
        0.1877815488033135,
        0.9765688961207617,
    )
    camera = compose_yaw_aligned_camera(quaternion, 0.37)
    expected = {
        "forward": (0.9939560979566968, 0.0, 0.10977830083717481),
        "left": (
            -0.019653562370291203,
            0.9838436927881214,
            0.17794753622118414,
        ),
        "up": (
            -0.10800468888365139,
            -0.17902957342582418,
            0.9778974378829883,
        ),
        # Analytic 0.326 * forward + 0.043 * up; mount left is zero.
        "origin": (
            0.3193854863118861,
            -0.007698271657310439,
            0.07783731590188749,
        ),
    }

    for actual, frozen in (
        (camera.forward_xyz, expected["forward"]),
        (camera.left_xyz, expected["left"]),
        (camera.up_xyz, expected["up"]),
        (camera.origin_xyz, expected["origin"]),
    ):
        assert len(actual) == len(frozen) == 3
        for actual_component, frozen_component in zip(actual, frozen):
            assert actual_component == pytest.approx(
                frozen_component, rel=0.0, abs=2e-15
            )


@pytest.mark.parametrize(
    ("quaternion_yaw", "stored_yaw"),
    (
        (0.0, 0.0),
        (0.37, 0.37),
        (-1.2, -1.2),
        (math.pi, -math.pi),
        (-math.pi, math.pi),
        (0.4, 0.4 + 2.0 * math.pi),
        (-0.7, -0.7 - 2.0 * math.pi),
        (1.1, 1.1 + 4.0 * math.pi),
    ),
)
def test_pure_yaw_is_invariant_under_wrapped_stored_yaws(
    quaternion_yaw: float,
    stored_yaw: float,
) -> None:
    quaternion = (
        0.0,
        0.0,
        math.sin(quaternion_yaw * 0.5),
        math.cos(quaternion_yaw * 0.5),
    )
    camera = compose_yaw_aligned_camera(quaternion, stored_yaw)
    mask = build_dynamic_cell_square_support_mask(quaternion, stored_yaw)

    assert camera.origin_xyz == pytest.approx((0.326, 0.0, 0.043), abs=2e-15)
    assert camera.forward_xyz == pytest.approx((1.0, 0.0, 0.0), abs=4e-15)
    assert camera.left_xyz == pytest.approx((0.0, 1.0, 0.0), abs=4e-15)
    assert camera.up_xyz == pytest.approx((0.0, 0.0, 1.0), abs=2e-15)
    assert _support_count(mask) == 2062
    assert support_mask_sha256(mask) == EXPECTED_LEVEL_SUPPORT_SHA256


@pytest.mark.parametrize(
    "norm",
    (
        math.nextafter(1.0 + 1e-5, 1.0),
        math.nextafter(1.0 - 1e-5, 1.0),
    ),
)
def test_quaternion_norm_tolerance_accepts_nearest_boundary_values(
    norm: float,
) -> None:
    camera = compose_yaw_aligned_camera((0.0, 0.0, 0.0, norm), 0.0)
    assert camera.forward_xyz == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "norm",
    (
        math.nextafter(1.0 + 1e-5, math.inf),
        math.nextafter(1.0 - 1e-5, 0.0),
    ),
)
def test_quaternion_norm_tolerance_rejects_just_outside_values(
    norm: float,
) -> None:
    with pytest.raises(ValueError, match="norm"):
        compose_yaw_aligned_camera((0.0, 0.0, 0.0, norm), 0.0)


@pytest.mark.parametrize("stored_yaw", (1e-5, -1e-5))
def test_yaw_tolerance_is_inclusive(stored_yaw: float) -> None:
    compose_yaw_aligned_camera((0.0, 0.0, 0.0, 1.0), stored_yaw)


@pytest.mark.parametrize(
    "stored_yaw",
    (math.nextafter(1e-5, math.inf), math.nextafter(-1e-5, -math.inf)),
)
def test_yaw_tolerance_rejects_one_ulp_outside(stored_yaw: float) -> None:
    with pytest.raises(ValueError, match="yaw"):
        compose_yaw_aligned_camera((0.0, 0.0, 0.0, 1.0), stored_yaw)


@pytest.mark.parametrize(
    ("quaternion", "inside_yaw", "outside_yaw"),
    (
        (
            (0.0, 0.0, 1.0, 0.0),
            math.nextafter(-math.pi + 1e-5, -math.inf),
            math.nextafter(-math.pi + 1e-5, math.inf),
        ),
        (
            (0.0, 0.0, -1.0, 0.0),
            math.nextafter(math.pi - 1e-5, math.inf),
            math.nextafter(math.pi - 1e-5, -math.inf),
        ),
    ),
)
def test_yaw_tolerance_wraps_across_both_pi_boundaries(
    quaternion: tuple[float, float, float, float],
    inside_yaw: float,
    outside_yaw: float,
) -> None:
    compose_yaw_aligned_camera(quaternion, inside_yaw)
    with pytest.raises(ValueError, match="yaw"):
        compose_yaw_aligned_camera(quaternion, outside_yaw)


def test_raw_quaternion_is_not_renormalized_during_composition() -> None:
    roll = 0.2
    unit = (math.sin(roll * 0.5), 0.0, 0.0, math.cos(roll * 0.5))
    scale = 1.0 + 4e-6
    scaled = tuple(value * scale for value in unit)

    unit_camera = compose_yaw_aligned_camera(unit, 0.0)
    scaled_camera = compose_yaw_aligned_camera(scaled, 0.0)

    assert scaled_camera.left_xyz != unit_camera.left_xyz
    assert scaled_camera.up_xyz != unit_camera.up_xyz


@pytest.mark.parametrize(
    ("cell", "quaternion", "stored_yaw", "expected_count", "expected_hash"),
    (
        (
            (22, 23),
            (
                0.01849944517016411,
                0.026065489277243614,
                -0.8357619643211365,
                -0.5481609106063843,
            ),
            1.9805201292037964,
            2082,
            "1c0baa473f4260f0911b378edd7f2cf3508580ddd0a4ad6243d9ed7b9b757d3a",
        ),
        (
            (17, 36),
            (
                -0.030500421300530434,
                -0.010054321028292179,
                -0.18749965727329254,
                0.9817396402359009,
            ),
            -0.3765529692173004,
            2079,
            "1a119ae0ad84625b1d9bcd968b9912c86b544c12902c35f1493ee2c36fefb4f7",
        ),
        (
            (22, 40),
            (
                -0.021598508581519127,
                -0.014208121225237846,
                0.6144306659698486,
                0.788547158241272,
            ),
            1.3237425088882446,
            2076,
            "b6bcbfeccdee5e251edb8ab4f7e9996a387867b68b7dcfd6fd983ac6a82673c1",
        ),
        (
            (22, 23),
            (
                0.023191479966044426,
                0.018481118604540825,
                -0.7698601484298706,
                -0.637523353099823,
            ),
            1.75795316696167,
            2078,
            "3ea1dd204e8618b29f43b226b2291220a611eac136fab40ef39da4d3a179e0ee",
        ),
    ),
)
def test_source_quaternion_boundary_cases_gain_dynamic_support(
    cell: tuple[int, int],
    quaternion: tuple[float, float, float, float],
    stored_yaw: float,
    expected_count: int,
    expected_hash: str,
) -> None:
    """Preserve source float values at three static-support boundary cells."""

    level_camera = compose_yaw_aligned_camera((0.0, 0.0, 0.0, 1.0), 0.0)
    dynamic_camera = compose_yaw_aligned_camera(quaternion, stored_yaw)

    assert cell_square_query_visible(*cell, level_camera) is False
    assert cell_square_query_visible(*cell, dynamic_camera) is True
    dynamic_mask = build_dynamic_cell_square_support_mask(quaternion, stored_yaw)
    assert _support_count(dynamic_mask) == expected_count
    assert support_mask_sha256(dynamic_mask) == expected_hash


def test_support_hash_rejects_shape_and_nonbool_ambiguity() -> None:
    level = build_dynamic_cell_square_support_mask((0.0, 0.0, 0.0, 1.0), 0.0)
    with pytest.raises(ValueError, match="shape"):
        support_mask_sha256(level[:-1])
    changed = [list(row) for row in level]
    changed[0][0] = 0
    with pytest.raises(TypeError, match="bool"):
        support_mask_sha256(changed)
