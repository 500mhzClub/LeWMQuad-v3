import numpy as np
import pytest

from lewm.batched_patch_tracker_development import chained_points
from lewm.cached_chain_association_development import CachedChainAssociation
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_chained_corner_flow_association_development import scene


def freeze(frames):
    for _, _, view in frames:
        view.gray = view.gray.copy()
        view.gray.setflags(write=False)
        view.depth = {k: a.copy() for k, a in view.depth.items()}
        for a in view.depth.values(): a.setflags(write=False)
    return frames


def same(a, b):
    assert a[1] == b[1]
    for x, y in zip(a[0], b[0], strict=True): np.testing.assert_array_equal(x, y)


def test_progressive_chains_reuse_links_without_changing_outputs_or_ownership():
    frames = freeze(scene(7)); cache = CachedChainAssociation()
    for end in range(2, 8): same(cache(frames[:end]), chained_points(frames[:end]))
    assert cache.hits == 15 and cache.misses == 6
    value = cache(frames)
    value[0][0][:] = 12345
    value[1]['steps'][0]['association']['counts']['photometric'] = -1
    same(cache(frames), chained_points(frames))


def test_writable_mutation_and_changed_reference_points_do_not_reuse_old_results():
    frames = scene(4, shift=0); cache = CachedChainAssociation()
    same(cache(frames), chained_points(frames))
    frames[1][2].gray.fill(0)
    same(cache(frames), chained_points(frames))
    assert cache.hits == 0 and len(cache(frames)[0][0]) == 0
    frames = freeze(scene(4)); same(cache(frames), chained_points(frames))
    frames[0][2].keypoints = frames[0][2].keypoints[::2]
    same(cache(frames), chained_points(frames))


def test_lost_tracks_stay_lost_and_clock_validation_remains_active_on_hits():
    frames = scene(5, shift=0)
    frames[2][2].depth = dict(depth_m=np.ones((480, 640)), valid=np.zeros((480, 640), bool))
    freeze(frames); cache = CachedChainAssociation()
    for end in range(2, 6): same(cache(frames[:end]), chained_points(frames[:end]))
    assert len(cache(frames)[0][0]) == 0 and cache.hits > 0
    bad = list(frames); bad[2] = (bad[2][0], bad[2][1]+1, bad[2][2])
    with pytest.raises(SensorContractError, match='100ms'): cache(bad)


def test_cache_is_bounded_and_expires_with_image_history():
    cache = CachedChainAssociation(); cache.maximum_entries = 2
    frames = freeze(scene(5)); same(cache(frames), chained_points(frames))
    assert len(cache.entries) == 2
    later = [(f+100, ns+10_000_000_000, image) for f, ns, image in freeze(scene(2))]
    same(cache(later), chained_points(later))
    assert len(cache.entries) == 1
