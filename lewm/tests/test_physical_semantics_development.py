"""Synthetic tests of the new assay; never initialize Genesis."""
import json
import math

import numpy as np
import pytest

from lewm.interface_semantics import InterfaceError, validate_rigid_transform
from lewm.physical_semantics import (
    blue_marker_centroid, evaluate_cases, project_world, world_from_optical,
)
from scripts import run_physical_semantics_development_v1 as RUN


def passing_cases():
    result = {}
    times = np.arange(401) * 0.002
    for name, color in (("clear", [0, 0, 0]), ("wall_red", [180, 0, 0]),
                        ("wall_green", [0, 180, 0])):
        x = 0.5 + times if name == "clear" else np.minimum(0.5 + times, 1.0)
        result[name] = {
            "rgb": np.broadcast_to(np.array(color, dtype=np.uint8), (480, 640, 3)).copy(),
            "positions_m": np.stack([x, np.zeros(401), np.full(401, 0.5)], axis=1).tolist(),
            "first_contact_step": None if name == "clear" else 250,
            "projection_error_px": [0.1, 0.2] if name == "clear" else [],
            "reset_position_error_m": 0.0, "reset_speed_mps": 0.0,
        }
    return result


def test_independent_optical_frame_is_proper_and_projects_left_up_correctly():
    transform = world_from_optical([0, 0, 0.5], [1, 0, 0], [0, 0, 1])
    validate_rigid_transform(transform)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)
    centre = project_world([2, 0, 0.5], transform)
    np.testing.assert_allclose(centre, [320, 240])
    point = project_world([2, 0.4, 0.7], transform)
    assert point[0] < 320 and point[1] < 240


def test_projection_yaw_matches_independent_angle_equation():
    yaw = math.radians(10)
    transform = world_from_optical([0, 0, 0], [math.cos(yaw), math.sin(yaw), 0], [0, 0, 1])
    pixel = project_world([2, 0.4, 0], transform)
    fx = 320 / math.tan(math.radians(78.323 / 2))
    assert pixel[0] == pytest.approx(320 - fx * math.tan(math.atan2(0.4, 2) - yaw))


@pytest.mark.parametrize("position,forward,up", [
    ([0, 0, 0], [0, 0, 0], [0, 0, 1]),
    ([0, 0, 0], [1, 0, 0], [2, 0, 0]),
    ([float("nan"), 0, 0], [1, 0, 0], [0, 0, 1]),
    ([0, 0], [1, 0, 0], [0, 0, 1]),
])
def test_optical_frame_rejects_degenerate_input(position, forward, up):
    with pytest.raises(InterfaceError):
        world_from_optical(position, forward, up)


def test_projection_rejects_behind_camera_and_reflections():
    transform = world_from_optical([0, 0, 0], [1, 0, 0], [0, 0, 1])
    with pytest.raises(InterfaceError):
        project_world([-1, 0, 0], transform)
    with pytest.raises(InterfaceError):
        project_world([1, 0, 0], np.diag([1, -1, 1, 1]))


def test_marker_centroid_is_measured_from_pixels_not_expected_projection():
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[3:7, 8:12, 2] = 200
    np.testing.assert_allclose(blue_marker_centroid(rgb), [10, 5])
    with pytest.raises(InterfaceError):
        blue_marker_centroid(np.zeros_like(rgb))


def test_assay_accepts_all_declared_positive_controls():
    report = evaluate_cases(passing_cases())
    assert report["status"] == "PASS" and all(report["checks"].values())


@pytest.mark.parametrize("case,field,bad", [
    ("clear", "first_contact_step", 250),
    ("wall_red", "first_contact_step", None),
    ("wall_red", "first_contact_step", 100),
    ("wall_green", "first_contact_step", 251),
    ("clear", "projection_error_px", [5, 0]),
    ("clear", "projection_error_px", [0, 5]),
    ("clear", "projection_error_px", [float("nan"), 0]),
    ("wall_red", "reset_position_error_m", 0.02),
    ("wall_green", "reset_speed_mps", 0.1),
])
def test_deliberate_physical_and_projection_failures_cannot_pass(case, field, bad):
    cases = passing_cases()
    cases[case][field] = bad
    assert evaluate_cases(cases)["status"] == "FAIL"


def test_invisible_collision_wall_cannot_pass():
    cases = passing_cases()
    cases["wall_red"]["rgb"] = cases["clear"]["rgb"].copy()
    assert not evaluate_cases(cases)["checks"]["visible_wall_changes_pixels"]


def test_material_that_changes_dynamics_cannot_pass():
    cases = passing_cases()
    cases["wall_green"]["positions_m"][-1][0] += 0.01
    assert not evaluate_cases(cases)["checks"]["material_invariant_dynamics"]


def test_truncated_physics_trace_fails_closed():
    cases = passing_cases()
    cases["clear"]["positions_m"] = cases["clear"]["positions_m"][:-1]
    with pytest.raises(InterfaceError):
        evaluate_cases(cases)


def test_runtime_failure_is_recorded_and_never_reported_as_pass(monkeypatch, tmp_path):
    def broken(*args):
        raise RuntimeError("synthetic renderer failure")
    monkeypatch.setattr(RUN, "collect_case", broken)
    output = tmp_path / "fresh"
    assert RUN.main(["--output-dir", str(output)]) == 1
    assert json.loads((output / "result.json").read_text())["status"] == "INFRASTRUCTURE_FAILURE"
    with pytest.raises(FileExistsError):
        RUN.main(["--output-dir", str(output)])
