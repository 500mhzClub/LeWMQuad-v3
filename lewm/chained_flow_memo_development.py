"""Reuse exact short-interval associations during a synchronous observation.

Only image/depth contents and original seed pixels identify a link. Cached
links still contain the original photometric and depth decisions. Mutable
camera arrays are checked again on every observation; returned arrays and
receipts never alias the cache. No pose or geometric fit is cached.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import numpy as np

from lewm.batched_patch_agreement_development import tracked_points as original
from lewm.direct_corner_flow_association_development import FLOW_RULES
from lewm.rgbd_correspondence_motion_development import RULES

_active = ContextVar('chained_flow_memo', default=None)


class ChainedFlowMemo:
    def __init__(self, *, image_capacity=128, link_capacity=2048):
        self.image_capacity = image_capacity
        self.link_capacity = link_capacity
        self.images = OrderedDict()
        self.links = OrderedDict()
        self.checked = {}
        self.serial = 0
        self.hits = self.misses = 0

    @contextmanager
    def observation(self):
        # The tracker consumes these inputs synchronously without mutating them.
        self.checked.clear()
        token = _active.set(self)
        try:
            yield
        finally:
            _active.reset(token)
            self.checked.clear()

    def image_token(self, view):
        arrays = (view.gray, view.depth['depth_m'], view.depth['valid'])
        key = tuple(id(a) for a in arrays)
        if key in self.checked:
            return self.checked[key]
        old = self.images.get(key)
        same = old is not None and all(
            a.shape == b.shape and a.dtype == b.dtype and a.strides == source.strides
            and np.array_equal(np.ascontiguousarray(a).view(np.uint8), b.view(np.uint8))
            for a, source, b in zip(arrays, old[1], old[2], strict=True))
        if not same:
            self.serial += 1
            # Retain source objects as well as private snapshots: IDs cannot be
            # recycled into an old entry while it remains in the table.
            old = (self.serial, arrays, tuple(np.ascontiguousarray(a).copy() for a in arrays))
            self.images[key] = old
        self.images.move_to_end(key)
        while len(self.images) > self.image_capacity:
            self.images.popitem(last=False)
        self.checked[key] = old[0]
        return old[0]

    def tracked_points(self, reference, current):
        seeds = tuple(tuple(k.pt) for k in reference.keypoints)
        key = (self.image_token(reference), self.image_token(current), seeds,
            tuple(sorted(FLOW_RULES.items())), tuple(sorted(RULES.items())))
        if key in self.links:
            self.hits += 1
            self.links.move_to_end(key)
            return deepcopy(self.links[key])
        self.misses += 1
        result = original(reference, current)
        self.links[key] = deepcopy(result)
        while len(self.links) > self.link_capacity:
            self.links.popitem(last=False)
        return result


def tracked_points(reference, current):
    memo = _active.get()
    return original(reference, current) if memo is None else memo.tracked_points(reference, current)
