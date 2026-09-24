import numpy as np
import pytest
from lewm.depth_boundary_counterexamples_development import fixture, PIXEL
from lewm.pixel_visible_surface_intervals_development import visible_intervals, supported_depth, area, subtract
from lewm.causal_depth_observation_development import FOCAL


def test_polygon_subtraction_preserves_area_and_keeps_hole_out():
    square = np.array([[0., 0.], [4., 0.], [4., 4.], [0., 4.]])
    hole = np.array([[1., 1.], [3., 1.], [3., 3.], [1., 3.]])
    pieces = subtract(square, hole)
    assert sum(abs(area(p)) for p in pieces) == pytest.approx(12.)
    assert subtract(square, square) == []


def test_small_region_rejects_missing_thin_post_and_fabricated_gap():
    T, boxes, _ = fixture()
    report = visible_intervals(boxes, T, PIXEL, radius_pixels=1/256)
    assert supported_depth(report, .96, metric_tolerance_m=.001)
    assert not supported_depth(report, 1.96, metric_tolerance_m=.001)
    assert not supported_depth(report, 1.5, metric_tolerance_m=.001)
    assert not report['supplied_angular_bound_validated']


def test_large_region_reports_both_physical_surfaces_without_filling_gap():
    T, boxes, _ = fixture()
    report = visible_intervals(boxes, T, PIXEL, radius_pixels=.5)
    assert supported_depth(report, .96, metric_tolerance_m=.001)
    assert supported_depth(report, 1.96, metric_tolerance_m=.001)
    assert not supported_depth(report, 1.5, metric_tolerance_m=.001)
    assert {p['object'] for p in report['intervals']} == {'thin_post', 'background'}
    assert sum(p['visible_projected_area_pixels2'] for p in report['intervals']) == pytest.approx(1.)


def test_physical_near_occluder_is_not_removed_by_native_clipping():
    T, boxes, _ = fixture(near=True)
    report = visible_intervals(boxes, T, PIXEL, radius_pixels=1/256)
    assert supported_depth(report, .004, metric_tolerance_m=0.)
    assert not supported_depth(report, 1.96, metric_tolerance_m=.001)


def test_order_invariance_and_absence_of_false_geometry():
    T, boxes, _ = fixture()
    a = visible_intervals(boxes, T, PIXEL, radius_pixels=.5)
    b = visible_intervals(list(reversed(boxes)), T, PIXEL, radius_pixels=.5)
    assert a == b
    assert not supported_depth(a, float('nan'), metric_tolerance_m=.001)
    assert not supported_depth(a, -.1, metric_tolerance_m=.001)


def ray_depth(boxes, transform, u, v):
    origin = transform[:3, 3]
    direction = transform[:3, :3]@np.array([(u-320)/FOCAL, (v-240)/FOCAL, 1.])
    result = -origin[2]/direction[2] if direction[2] < 0 else np.inf
    for box in boxes:
        c, s = np.cos(box['yaw_rad']), np.sin(box['yaw_rad'])
        R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        p, d, h = (origin-box['centre_xyz'])@R, direction@R, np.asarray(box['size_xyz'])/2
        low, high = 0., np.inf
        for axis in range(3):
            if abs(d[axis]) < 1e-14:
                if abs(p[axis]) > h[axis]: high = -1.; break
            else:
                a, b = (-h[axis]-p[axis])/d[axis], (h[axis]-p[axis])/d[axis]
                low, high = max(low, min(a, b)), min(high, max(a, b))
        if 0 < low <= high: result = min(result, low)
    return result


def test_rotated_overlapping_faces_cover_independently_traced_interior_rays():
    rng = np.random.default_rng(4499)
    for _ in range(8):
        T, boxes, _ = fixture()
        boxes[1]['yaw_rad'] = float(rng.uniform(-.9, .9))
        boxes[1]['centre_xyz'][1] += float(rng.uniform(-.002, .002))
        boxes.append(dict(wall_id='second_post', centre_xyz=[1.3, -.014, .6],
            size_xyz=[.05, .001, 1.], yaw_rad=float(rng.uniform(-.9, .9))))
        report = visible_intervals(boxes, T, PIXEL, radius_pixels=.5)
        for dv in np.linspace(-.49, .49, 7):
            for du in np.linspace(-.49, .49, 7):
                d = ray_depth(boxes, T, PIXEL[1]+.5+du, PIXEL[0]+.5+dv)
                assert supported_depth(report, d, metric_tolerance_m=1e-10)


def test_occluded_background_is_rejected_for_a_broad_rotated_foreground():
    T, boxes, _ = fixture()
    boxes[1]['size_xyz'][1] = .2; boxes[1]['yaw_rad'] = .4
    report = visible_intervals(boxes, T, PIXEL, radius_pixels=.5)
    assert not supported_depth(report, 1.96, metric_tolerance_m=.001)
    assert {p['object'] for p in report['intervals']} == {'thin_post'}


@pytest.mark.parametrize('radius', [0., -.1, .51, float('nan')])
def test_unsupported_region_is_rejected(radius):
    T, boxes, _ = fixture()
    with pytest.raises(ValueError): visible_intervals(boxes, T, PIXEL, radius_pixels=radius)
