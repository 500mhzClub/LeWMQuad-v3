"""Compare complete candidate populations with the original dense geometry."""
import numpy as np
import pytest

from lewm import sampled_plane_candidates_development as candidate
from lewm.floor_pose_registration_development import measured_candidates as original
from lewm.tests.test_floor_pose_registration_development import render_plane
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL


@pytest.mark.parametrize('mount', [np.asarray(BODY_FROM_OPTICAL), body_from_optical()])
@pytest.mark.parametrize('kind', ['plane', 'missing', 'noise', 'step', 'random', 'empty', 'strided'])
def test_exact_candidates_and_masks(mount, kind):
    depth, valid = render_plane(mount, np.array([0., 0., 1.]), .32)
    rng = np.random.default_rng(20260913)
    if kind == 'missing':
        valid[rng.random(valid.shape) < .2] = False; depth[~valid] = 0.
    elif kind == 'noise':
        depth[valid] = np.clip(depth[valid]+rng.normal(0, .002, valid.sum()), .2, 5.)
    elif kind == 'step':
        depth[:, 320:] = np.where(valid[:, 320:], np.clip(depth[:, 320:]+.1, .2, 5.), 0.)
    elif kind == 'random':
        depth = rng.uniform(.2, 5., depth.shape); valid[:] = True
    elif kind == 'empty':
        depth[:] = 0.; valid[:] = False
    elif kind == 'strided':
        depth = depth[:, ::-1]; valid = valid[:, ::-1]
    before = depth.tobytes(), valid.tobytes()
    for up in (np.array([0., 0., 1.]), np.array([.1, -.1, np.sqrt(.98)])):
        old = original(depth, valid, mount, up)
        new = candidate.measured_candidates(depth, valid, mount, up)
        for a, b in zip(old, new, strict=True):
            assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert before == (depth.tobytes(), valid.tobytes())


@pytest.mark.parametrize('delta', [-1e-10, -np.finfo(float).eps, 0., np.finfo(float).eps, 1e-10])
def test_floor_alignment_boundary(delta):
    mount = np.asarray(BODY_FROM_OPTICAL)
    depth, valid = render_plane(mount, np.array([0., 0., 1.]), .32)
    z = .97+delta; up = np.array([np.sqrt(1-z*z), 0., z])
    for a, b in zip(original(depth, valid, mount, up),
            candidate.measured_candidates(depth, valid, mount, up), strict=True):
        np.testing.assert_array_equal(a, b)


def test_unsampled_invalid_input_still_rejected():
    mount = np.asarray(BODY_FROM_OPTICAL)
    depth, valid = render_plane(mount, np.array([0., 0., 1.]), .32)
    depth[0, 0] = np.nan
    for fn in (original, candidate.measured_candidates):
        with pytest.raises(ValueError):
            fn(depth, valid, mount, [0., 0., 1.])
