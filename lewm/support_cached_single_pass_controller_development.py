"""Same combined controller with a fresh support cache per contact query."""
from lewm.scoped_support_query_cache_development import SupportQueryCache
from lewm.single_pass_receipt_copied_controller_development import (
    SinglePassMeasuredFloorMemory, SinglePassMeasuredFloorMap, SinglePassReceiptCopiedController)


class SupportCachedSinglePassMemory(SinglePassMeasuredFloorMemory):
    def footprint(self, geometry, displacement_body_xy, yaw_rad, *, now_ns, persistent=True):
        cache = SupportQueryCache(geometry)
        try:
            with cache:
                return super().footprint(cache, displacement_body_xy, yaw_rad, now_ns=now_ns, persistent=persistent)
        finally:
            self.last_support_query_counts = cache.counts()


class SupportCachedSinglePassMap(SinglePassMeasuredFloorMap):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.surface = SupportCachedSinglePassMemory(identity=identity)


class SupportCachedSinglePassController(SinglePassReceiptCopiedController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = SupportCachedSinglePassMap(identity=(0, 0, 0))
        self.memory = self.mapper.surface
