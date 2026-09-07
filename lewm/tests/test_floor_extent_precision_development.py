from copy import deepcopy

import numpy as np
import pytest

from lewm.tests.test_aligned_floor_development import identity
from lewm_genesis.floor_extent_precision_development import (
    EXTENTS_M, VIEWS, check_extent_identity, decode_depth64, decode_native_reference, evaluate, reference,
)


@pytest.mark.parametrize('extent', EXTENTS_M)
def test_actual_extent_and_surface_roles(extent):
    row = identity()
    v = np.asarray(row['visual_local_vertices_m']); v[:, :2] *= extent / 1000.
    row['visual_local_vertices_m'] = v.tolist()
    report = check_extent_identity(row, extent)
    assert report['scene_surface_alignment_verified']
    assert report['actual_visual_extent_m'] == extent
    bad = deepcopy(row); bad['visual_position_world_m'][2] = 0.
    with pytest.raises(ValueError): check_extent_identity(bad, extent)
    bad = deepcopy(row); bad['visual_local_vertices_m'][0][0] += .01
    with pytest.raises(ValueError): check_extent_identity(bad, extent)


@pytest.mark.parametrize('view', VIEWS)
def test_all_new_view_rays_have_matched_visual_support(view):
    large, use = reference(view, 1000.)
    small, use2 = reference(view, 32.)
    np.testing.assert_array_equal(large, small); np.testing.assert_array_equal(use, use2)
    assert use.sum() > 1000
    # Construct an independently quantized ideal 24-bit buffer, not native data.
    d = np.where(use, large, 200.)
    z = ((200. + .05 - 2 * .05 * 200. / d) / (200. - .05) + 1) / 2
    raw = (np.round(z * (2**24 - 1)) / (2**24 - 1)).astype(np.float32)
    native = decode_native_reference(raw, np.float32(.05), np.float32(200.))
    report = evaluate(view, 32., native, raw)
    assert report['native_within1mm'] and report['native_buffer_reconstruction_exact']
    native[use] += .002
    report = evaluate(view, 32., native, raw)
    assert not report['native_within1mm'] and not report['native_buffer_reconstruction_exact']


@pytest.mark.parametrize('fault', ['nan', 'negative', 'over', 'dtype', 'shape', 'clip'])
def test_depth_buffer_contract(fault):
    raw = np.ones((480, 640), np.float32)
    if fault == 'nan': raw[0, 0] = np.nan
    if fault == 'negative': raw[0, 0] = -.01
    if fault == 'over': raw[0, 0] = 1.01
    if fault == 'dtype': raw = raw.astype(np.float64)
    if fault == 'shape': raw = raw[:1]
    with pytest.raises(ValueError): decode_depth64(raw, far=100. if fault == 'clip' else 200.)


def test_finite_floor_boundary_cannot_be_extrapolated():
    with pytest.raises(ValueError): reference((20., 0., .3, 0., 0.), 32.)
