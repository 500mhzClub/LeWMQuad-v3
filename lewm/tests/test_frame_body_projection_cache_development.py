"""Byte-exact mapping arithmetic, bounded ownership and no cross-frame reuse."""
import ast
import inspect

import numpy as np
import pytest

from lewm import frame_body_projection_cache_development as candidate
from lewm import frame_cached_floor_geometry_development as original
from lewm.current_primary_floor_plane_development import measured_points


def original_body(depth):
    yy, xx = np.indices((480, 640))
    optical = np.stack((depth*(xx+.5-320)/original.FOCAL,
        depth*(yy+.5-240)/original.FOCAL, depth), axis=-1)
    return optical@original.T[:3, :3].T+original.T[:3, 3]


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.uint16, np.bool_])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'fortran'])
def test_projection_bytes_match_original_mapping_expression(dtype, layout):
    data = np.random.default_rng(19).uniform(.2, 5., size=(480, 1280)).astype(dtype)
    depth = data[:, ::2] if layout == 'strided' else data[:, :640].copy(order='F' if layout == 'fortran' else 'C')
    before = depth.tobytes(); expected = original_body(depth)
    cache = candidate.FrameBodyProjectionCache(); actual = cache.body(depth)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes() and depth.tobytes() == before
    assert cache.body(depth.copy()) is actual and cache.counts() == dict(hits=1, misses=1, uncached=0)
    with pytest.raises(ValueError): actual.flags.writeable = True
    with pytest.raises(ValueError): actual[0, 0, 0] = 1.


def test_projection_retains_exact_arithmetic_of_all_four_mapping_consumers():
    import textwrap
    expected = ast.parse('np.stack((depth*(xx+.5-320)/FOCAL, depth*(yy+.5-240)/FOCAL, depth), axis=-1)', mode='eval').body
    for function in (original.FloorFrameGeometry.floor_coverage, original.FloorFrameGeometry.sampled_floor_patch,
            original.FloorFrameGeometry.append_patch, measured_points, candidate.project_body):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'optical' for target in node.targets)]
        assert len(assignments) == 1 and ast.dump(assignments[0].value) == ast.dump(expected)


def test_signed_zero_and_small_input_changes_never_share_cached_projection():
    depth = np.ones((480, 640)); depth[0, 0] = 0.
    cache = candidate.FrameBodyProjectionCache(); first = cache.body(depth)
    depth[0, 0] = -.0; second = cache.body(depth)
    assert first is not second and cache.counts()['misses'] == 2
    depth[0, 1] = np.nextafter(1., 2.); third = cache.body(depth)
    assert third.tobytes() == original_body(depth).tobytes()
    assert cache.counts() == dict(hits=0, misses=3, uncached=1)


def test_source_mutation_cannot_change_previously_cached_projection():
    depth = np.ones((480, 640)); cache = candidate.FrameBodyProjectionCache()
    first = cache.body(depth); before = first.tobytes()
    depth[100, 200] = 2.; second = cache.body(depth)
    assert first.tobytes() == before and second is not first
    assert second.tobytes() == original_body(depth).tobytes()


def test_capacity_overflow_recomputes_and_close_clears_scope():
    cache = candidate.FrameBodyProjectionCache()
    for value in (1., 2., 3., 3.): cache.body(np.full((480, 640), value))
    assert len(cache._entries) == 2 and cache.counts() == dict(hits=0, misses=4, uncached=2)
    cache.close(); assert cache.closed and cache._entries == {}
    with pytest.raises(ValueError, match='closed'): cache.body(np.ones((480, 640)))
    next_frame = candidate.FrameBodyProjectionCache(); next_frame.body(np.ones((480, 640)))
    assert next_frame.counts() == dict(hits=0, misses=1, uncached=0)


def test_camera_transform_changes_are_part_of_exact_cache_identity(monkeypatch):
    depth = np.ones((480, 640)); cache = candidate.FrameBodyProjectionCache()
    first = cache.body(depth)
    transform = np.asarray(candidate.BODY_FROM_OPTICAL).copy(); transform[0, 3] += .01
    monkeypatch.setattr(candidate, 'BODY_FROM_OPTICAL', transform)
    second = cache.body(depth)
    assert second is not first and second.tobytes() == candidate.project_body(depth).tobytes()
    assert cache.counts()['misses'] == 2


@pytest.mark.parametrize('shape', [(2, 2), (480, 640, 1)])
def test_malformed_shape_follows_original_arithmetic_failure_without_caching(shape):
    depth = np.ones(shape); cache = candidate.FrameBodyProjectionCache()
    with pytest.raises(ValueError): original_body(depth)
    with pytest.raises(ValueError): cache.body(depth)
    assert cache._entries == {} and cache.counts()['uncached'] == 1


def test_array_subclass_uses_uncached_original_expression():
    class Depth(np.ndarray): pass
    depth = np.ones((480, 640)).view(Depth)
    cache = candidate.FrameBodyProjectionCache()
    for _ in range(2): assert cache.body(depth).tobytes() == original_body(depth).tobytes()
    assert cache.counts() == dict(hits=0, misses=0, uncached=2) and cache._entries == {}
