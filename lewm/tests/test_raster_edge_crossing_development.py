import numpy as np
import pytest
from lewm.raster_edge_crossing_development import OFFSETS_PIXELS, poses, score_frame
from lewm.physical_first_surface_depth_development import expected_optical_depth
from lewm.tests.test_raster_footprint_visibility_development import fixture


def test_fixed_symmetric_subpixel_and_whole_pixel_population():
    _, T, _, _ = fixture()
    rows = poses(T)
    assert len(rows) == 9 and list(OFFSETS_PIXELS) == [-v for v in reversed(OFFSETS_PIXELS)]
    np.testing.assert_array_equal(rows[4]['world_from_optical'], T)
    for row in rows:
        R = np.array(row['world_from_optical'])
        np.testing.assert_array_equal(R[:3, 3], T[:3, 3])
        np.testing.assert_allclose(R[:3, :3].T @ R[:3, :3], np.eye(3), atol=1e-12)
        assert np.linalg.det(R[:3, :3]) > .999999


@pytest.mark.parametrize('offset_index', range(9))
def test_all_perturbed_analytic_views_reject_prescribed_interior_error(offset_index):
    boxes, T, _, _ = fixture()
    T = poses(T)[offset_index]['world_from_optical']
    ref = expected_optical_depth(boxes, T)
    native = np.zeros((480, 640), np.float32)
    native[np.ix_(ref['rows'], ref['columns'])] = ref['expected_depth_m']
    original = native.copy()
    result = score_frame(native, boxes, T)
    np.testing.assert_array_equal(native, original)
    assert result['footprint']['stable_interior_metric_pass']
    assert result['interior_negative_control']['rejected']
    assert result['interior_negative_control']['stable_bad_rays'] == 1
    assert not result['boundary_pixels_certified'] and not result['policy_filter']


def test_boundary_failure_is_retained_not_rescored_as_strict_pass():
    boxes, T, native, ref = fixture()
    native[ref['rows'][30], ref['columns'][40]] = 1.
    result = score_frame(native, boxes, T)
    assert result['footprint']['boundary_bad_rays'] == 1
    assert not result['footprint']['original_strict_score']['passes_sampled_physical_visibility']
    assert result['interior_negative_control']['rejected']


@pytest.mark.parametrize('fault', ['reflection', 'scale', 'nan', 'last_row'])
def test_invalid_poses_are_rejected(fault):
    _, T, _, _ = fixture()
    if fault == 'reflection': T[:3, 0] *= -1
    elif fault == 'scale': T[:3, 0] *= 2
    elif fault == 'nan': T[0, 3] = np.nan
    else: T[3, 0] = 1
    with pytest.raises(ValueError): poses(T)
