"""Bounded reuse of local depth estimates for owned retained camera arrays."""
from collections import OrderedDict
from lewm.local_feature_depth_consensus_development import (
    LocalFeatureDepthConsensusPose, _observation_cache)
from lewm.gyro_seeded_height_floor_tracking_development import (
    GyroSeededHeightFloorPose, GyroSeededHeightFloorMotion)


class RetainedDepthCache(OrderedDict):
    """Keep exact input identities alive and make owned cached arrays read-only.

    Feature frames own deep copies of sensor arrays. Non-owning inputs remain
    uncached because a writable backing array could invalidate their values.
    Only the unchanged local depth calculation is reused, never a pose result.
    """
    def __init__(self, maximum_entries=32):
        super().__init__()
        if type(maximum_entries) is not int or not 1 <= maximum_entries <= 32:
            raise ValueError('bounded positive cache size required')
        self.maximum_entries=maximum_entries;self.hits=0;self.misses=0

    def get(self,key,default=None):
        if key not in self:
            self.misses+=1;return default
        entry=super().__getitem__(key)
        if any(a.flags.writeable for a in entry[0]):
            raise ValueError('retained depth input became writable')
        self.hits+=1;self.move_to_end(key);return entry

    def __setitem__(self,key,entry):
        arrays,estimated=entry
        if not all(a.flags.owndata for a in arrays):return
        if key!=tuple(id(a) for a in arrays):
            raise ValueError('exact owned array identities required')
        for a in (*arrays,*estimated.values()):a.setflags(write=False)
        super().__setitem__(key,entry);self.move_to_end(key)
        while len(self)>self.maximum_entries:self.popitem(last=False)


class RetainedDepthCachePose(GyroSeededHeightFloorPose):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.retained_depth_cache=RetainedDepthCache()

    def observe(self,*args,**kwargs):
        token=_observation_cache.set(self.retained_depth_cache)
        try:
            # Replace only LocalFeatureDepthConsensusPose's per-call cache scope.
            # Its immediate superclass receives the same call and unchanged MRO.
            return super(LocalFeatureDepthConsensusPose,self).observe(*args,**kwargs)
        finally:
            _observation_cache.reset(token)


class RetainedDepthCacheMotion(GyroSeededHeightFloorMotion):
    def __init__(self,*,identity=(0,0,0),activation_frame=0):
        super().__init__(identity=identity,activation_frame=activation_frame)
        self.model=RetainedDepthCachePose(activation_frame=activation_frame)

    def snapshot(self,*,now_ns):
        return super().snapshot(now_ns=now_ns)|dict(
            retained_local_depth_cache=True,retained_local_depth_cache_maximum_entries=32,
            local_depth_estimator_and_pose_acceptance_unchanged=True)
