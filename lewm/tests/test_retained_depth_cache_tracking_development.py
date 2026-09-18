import numpy as np
import pytest
from lewm.local_feature_depth_consensus_development import lift, _observation_cache
from lewm.retained_depth_cache_tracking_development import RetainedDepthCache


def call(cache,depth):
    token=_observation_cache.set(cache)
    try:return lift(depth,np.array([[320.1,300.4],[319.2,299.3]]))
    finally:_observation_cache.reset(token)


def packet(value=1.):
    return dict(depth_m=np.full((480,640),value,np.float32),valid=np.ones((480,640),bool))


def test_reuse_preserves_outputs_inputs_and_rejects_writable_transition():
    depth=packet();depth['valid'][300,321]=False;depth['depth_m'][300,321]=0
    original={k:v.copy() for k,v in depth.items()}
    expected=call({},depth);cache=RetainedDepthCache()
    for _ in range(2):
        actual=call(cache,depth)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    assert cache.hits==1 and cache.misses==1
    for k,v in depth.items():
        np.testing.assert_array_equal(v,original[k]);assert not v.flags.writeable
    depth['depth_m'].setflags(write=True)
    with pytest.raises(ValueError,match='became writable'):call(cache,depth)


def test_lru_bound_and_no_stale_nonowned_cache():
    cache=RetainedDepthCache(2);a,b,c=packet(),packet(1.1),packet(1.2)
    call(cache,a);call(cache,b);call(cache,a);call(cache,c)
    assert len(cache)==2 and tuple(id(v) for v in b.values()) not in cache
    backing=packet();view={k:v.view() for k,v in backing.items()}
    first=call(cache,view);backing['depth_m'][:]=1.5;second=call(cache,view)
    assert not np.array_equal(first[0],second[0])
    assert tuple(id(v) for v in view.values()) not in cache
