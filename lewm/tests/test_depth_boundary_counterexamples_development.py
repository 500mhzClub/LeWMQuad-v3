import numpy as np
from lewm.depth_boundary_counterexamples_development import diagnose, fixture, PIXEL
from lewm.physical_first_surface_depth_development import expected_optical_depth


def test_thin_missing_and_fabricated_depth_escape_both_interior_metrics():
    report = diagnose()
    for row in report['cases'][:2]:
        assert row['sampled_foreground_rays'] > 0
        assert row['tracked_expected_depth_m'] == .96
        assert row['tracked_synthetic_depth_m'] > 1.4
        assert row['original_strict_score']['passes_sampled_physical_visibility']
        assert row['footprint_diagnostic']['stable_interior_bad_rays'] == 0
    assert report['cases'][1]['tracked_synthetic_depth_m'] == 1.5
    assert not report['boundary_accounting_alone_sufficient_for_visibility']


def test_near_occluder_remains_a_failure_despite_thin_surface_exclusion():
    row = diagnose()['cases'][2]
    assert row['tracked_expected_depth_m'] < .005
    assert not row['tracked_expected_surface_interior']
    assert row['original_strict_score']['clipped_opaque_rays'] > 0
    assert row['original_strict_score']['false_public_valid_near_rays'] > 0


def test_fabricated_point_is_in_empty_space_even_inside_one_pixel_footprint():
    transform, boxes, background = fixture()
    # At optical depth 1.5 m, the complete one-pixel footprint has world x=1.5.
    # It is between the foreground's back x=1.04 and background's front x=1.96,
    # above the ground. Thus no permitted horizontal sample displacement makes
    # this fabricated value a physical surface return.
    assert boxes[1]['centre_xyz'][0]+boxes[1]['size_xyz'][0]/2 < 1.5
    assert boxes[0]['centre_xyz'][0]-boxes[0]['size_xyz'][0]/2 > 1.5
    assert background[PIXEL] > 1.9
    a = expected_optical_depth(boxes, transform, stride=1)
    b = expected_optical_depth(list(reversed(boxes)), transform, stride=1)
    np.testing.assert_array_equal(a['expected_depth_m'], b['expected_depth_m'])
