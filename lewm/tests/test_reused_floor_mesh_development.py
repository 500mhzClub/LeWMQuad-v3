import numpy as np
import pytest

from lewm.floor_footprint_bounds_development import observed_floor_cell_index
from lewm.frame_floor_index_cache_development import FrameFloorIndexCache
from lewm.reused_floor_mesh_development import ReusedFloorMeshCache
from lewm.tests.test_current_primary_floor_plane_development import depth_plane


def equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        a, b = left[name], right[name]
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.tobytes() == b.tobytes(), name


@pytest.mark.parametrize('scene', ['plane', 'height_boundary', 'slope_boundary',
    'wall', 'empty', 'noisy', 'missing_neighbors'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_complete_mesh_outputs_match_original_for_distinct_exact_up_vectors(scene, dtype):
    d, v = depth_plane()
    if scene == 'height_boundary':
        d, v = depth_plane(height=np.nextafter(-.15, -np.inf), slope=0.)
    elif scene == 'slope_boundary':
        d, v = depth_plane(slope=np.sqrt(1/.97**2-1))
    elif scene == 'wall':
        d = np.ones_like(d); v = np.ones_like(v)
    elif scene == 'empty':
        d = np.zeros_like(d); v = np.zeros_like(v)
    elif scene == 'noisy':
        rng = np.random.default_rng(20260910)
        d = rng.uniform(.2, 5., size=d.shape); v = np.ones_like(v)
    elif scene == 'missing_neighbors':
        d[::7, ::11] = 0.; v[::7, ::11] = False
    d = d.astype(dtype)
    cache = ReusedFloorMeshCache()
    for tilt in [0., 1e-12, .01, -.02]:
        u = np.array([tilt, 0., 1.]); u /= np.linalg.norm(u)
        equal(cache.index(d, v, u), observed_floor_cell_index(d, v, u))
    assert len(cache._meshes) == 1 and len(cache._entries) == 4
    assert cache.counts() == dict(hits=0, misses=4, uncached=0)


def test_exact_up_key_and_owned_immutable_results_survive_input_mutation():
    d, v = depth_plane(); cache = ReusedFloorMeshCache()
    old = cache.index(d, v, [0., 0., 1.]); saved = {k:a.copy() for k,a in old.items()}
    assert cache.index(d.copy(), v.copy(), [0., 0., 1.]) is old
    for a in old.values():
        with pytest.raises(ValueError):
            a.flags.writeable = True
    with pytest.raises(TypeError):
        old['up'] = np.zeros(3)
    r, c = np.argwhere(v)[100]
    d[r, c] = 0.; v[r, c] = False
    equal(cache.index(d, v, [0., 0., 1.]), observed_floor_cell_index(d, v, [0., 0., 1.]))
    equal(old, saved)
    assert len(cache._meshes) == 2


@pytest.mark.parametrize('fault', ['nan', 'invalid_nonzero', 'range', 'up_norm',
    'up_nan', 'up_shape', 'shape', 'valid_dtype', 'object_depth'])
def test_original_rejections_after_a_successful_cache_hit(fault):
    d, v = depth_plane(); u = np.array([0., 0., 1.])
    cache = ReusedFloorMeshCache(); cache.index(d, v, u); cache.index(d, v, u)
    if fault == 'nan': d[0, 0] = np.nan
    elif fault == 'invalid_nonzero': d[0, 0] = 1.; v[0, 0] = False
    elif fault == 'range': d[0, 0] = 5.01; v[0, 0] = True
    elif fault == 'up_norm': u[2] = 2.
    elif fault == 'up_nan': u[2] = np.nan
    elif fault == 'up_shape': u = u[:2]
    elif fault == 'shape': d = d[:10]
    elif fault == 'valid_dtype': v = v.astype(int)
    else: d = d.astype(object)
    with pytest.raises(Exception) as original:
        observed_floor_cell_index(d, v, u)
    with pytest.raises(type(original.value)) as candidate:
        cache.index(d, v, u)
    assert str(candidate.value) == str(original.value)
    assert len(cache._meshes) == len(cache._entries) == 1


def test_dtype_camera_and_logical_array_layout_keys_preserve_original_semantics():
    d, v = depth_plane(); old = FrameFloorIndexCache(); new = ReusedFloorMeshCache()
    other, ov = depth_plane(height=-.35)
    for depth, valid in [(d,v), (np.asfortranarray(d), np.asfortranarray(v)),
                         (d.astype('>f4'),v), (d.astype(np.float64),v), (other,ov)]:
        equal(old.index(depth, valid, [0., 0., 1.]), new.index(depth, valid, [0., 0., 1.]))
    assert old.counts() == new.counts() == dict(hits=1, misses=4, uncached=0)
    assert len(new._meshes) == 4


def test_both_caches_are_bounded_and_cleared_at_observation_end():
    old = FrameFloorIndexCache(); new = ReusedFloorMeshCache()
    for i in range(10):
        d, v = depth_plane(height=-.3-i*.001)
        equal(old.index(d,v,[0.,0.,1.]), new.index(d,v,[0.,0.,1.]))
    assert old.counts() == new.counts() == dict(hits=0, misses=10, uncached=2)
    assert len(new._meshes) == len(new._entries) == 8
    new.close()
    assert not new._meshes and not new._entries and new.closed
    with pytest.raises(ValueError, match='scope already closed'):
        new.index(d,v,[0.,0.,1.])
