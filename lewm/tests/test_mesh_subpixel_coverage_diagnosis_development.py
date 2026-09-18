import numpy as np
import pytest
from lewm.causal_depth_observation_development import FOCAL
from lewm.mesh_subpixel_coverage_diagnosis_development import diagnose_mesh, barycentric


def fixture():
    uv = np.array([[320.499, 239.5], [320.499, 241.5], [318.5, 240.5],
        [319., 239.], [322., 239.], [320.5, 243.]])
    z = np.array([1., 1., 1., 2., 2., 2.])
    vertices = np.c_[(uv-[320., 240.])/FOCAL*z[:, None], z].astype(np.float32)
    return vertices, np.array([[0, 1, 2], [3, 4, 5]])


def test_grid_snapping_can_change_covering_triangle_without_rewriting_depth():
    v, f = fixture(); before = v.copy()
    r = diagnose_mesh(v, f, np.eye(4), [240, 320], subpixel_bits=8, near_m=.005)
    assert r['exact_mesh_projection']['nearest_candidates'][0]['triangle_index'] == 1
    hit = r['hypothetical_snapped_projection']['nearest_candidates'][0]
    assert hit['triangle_index'] == 0 and hit['unsnapped_plane_depth_m'] == pytest.approx(1.)
    assert hit['unsnapped_plane_value_may_extrapolate_outside_triangle']
    np.testing.assert_array_equal(v, before)
    assert not r['native_clipping_shader_arithmetic_and_culling_reproduced']


def test_degenerate_or_near_crossing_triangles_do_not_masquerade_as_complete_raster():
    v, f = fixture(); v[0, 2] = .001
    r = diagnose_mesh(v, np.r_[f, [[3, 3, 3]]], np.eye(4), [240, 320], subpixel_bits=8, near_m=.005)
    assert r['excluded_triangles'] == 1
    assert r['hypothetical_snapped_projection']['covering_triangles'] == 1
    assert not r['complete_visibility_evaluation']


@pytest.mark.parametrize('bad', ['dtype', 'face', 'pose', 'pixel', 'bits'])
def test_invalid_mesh_evidence_rejected(bad):
    v, f = fixture(); T = np.eye(4); pixel = [240, 320]; bits = 8
    if bad == 'dtype': v = v.astype(float)
    elif bad == 'face': f[0, 0] = 99
    elif bad == 'pose': T[0, 0] = -1.
    elif bad == 'pixel': pixel = [480, 320]
    else: bits = 0
    with pytest.raises(ValueError): diagnose_mesh(v, f, T, pixel, subpixel_bits=bits, near_m=.005)
