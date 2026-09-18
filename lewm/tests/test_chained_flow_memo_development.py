import numpy as np
import pytest
from lewm.batched_patch_tracker_development import chained_points
from lewm.chained_flow_memo_development import ChainedFlowMemo
from lewm.direct_corner_flow_association_development import FLOW_RULES
from lewm.tests.test_chained_corner_flow_association_development import scene


def identical(a, b):
    assert a[1] == b[1]
    for x, y in zip(a[0], b[0], strict=True):
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def test_overlapping_chains_reuse_exact_links_and_isolate_returned_results():
    frames = scene(6)
    memo = ChainedFlowMemo()
    for length in (3, 5, 6):
        expected = chained_points(frames[:length])
        with memo.observation():
            result = chained_points(frames[:length])
            identical(result, expected)
            result[0][0].fill(123)
            result[1]['steps'][0]['association']['counts']['photometric'] = -1
            identical(chained_points(frames[:length]), expected)
    assert memo.misses == 5 and memo.hits > 5


@pytest.mark.parametrize('change', ['image', 'depth', 'valid', 'seeds', 'rule'])
def test_changed_inputs_or_rules_recompute_original_decisions(change, monkeypatch):
    frames = scene(4, shift=0)
    memo = ChainedFlowMemo()
    with memo.observation():
        chained_points(frames)
    misses = memo.misses
    if change == 'image': frames[1][2].gray.fill(0)
    elif change == 'depth': frames[1][2].depth['depth_m'] *= 1.5
    elif change == 'valid': frames[1][2].depth['valid'].fill(False)
    elif change == 'seeds': frames[0][2].keypoints = frames[0][2].keypoints[:10]
    else: monkeypatch.setitem(FLOW_RULES, 'minimum_patch_zncc', 1.01)
    expected = chained_points(frames)
    with memo.observation():
        identical(chained_points(frames), expected)
    assert memo.misses > misses
    if change in ('image', 'valid', 'rule'):
        assert len(expected[0][0]) == 0


def test_eviction_changes_cost_only():
    frames = scene(6)
    memo = ChainedFlowMemo(image_capacity=2, link_capacity=2)
    expected = chained_points(frames)
    for _ in range(2):
        with memo.observation():
            identical(chained_points(frames), expected)
        assert len(memo.images) <= 2 and len(memo.links) <= 2
