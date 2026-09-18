"""Reuse exact image-link results within the existing 3.2-second history.

Only privately owned, read-only image/depth arrays are eligible. The original
chain algorithm still validates clocks, preserves original pixel identities,
drops lost tracks, and lifts both endpoints. No pose or fit is cached.
"""
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from lewm.batched_patch_agreement_development import tracked_points
from lewm.chained_corner_flow_association_development import chained_points, CHAIN_RULES
from lewm.eligible_floor_registration_development import bind


class CachedChainAssociation:
    def __init__(self):
        self.entries = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.maximum_entries = 256

    def __call__(self, frames):
        # Malformed chains go directly to the original validator.
        if not isinstance(frames, (list, tuple)) or not frames:
            return chained_points(frames)
        if any(not isinstance(r, tuple) or len(r) != 3 or type(r[1]) is not int for r in frames):
            return chained_points(frames)
        now = frames[-1][1]
        oldest = now-CHAIN_RULES['maximum_intervals']*CHAIN_RULES['sample_interval_ns']
        self.entries = OrderedDict((k, v) for k, v in self.entries.items() if v[0] >= oldest)
        clocks = {id(r[2].gray): r[1] for r in frames}

        def cached_link(reference, current):
            arrays = tuple(a for f in (reference, current)
                for a in (f.gray, f.depth['depth_m'], f.depth['valid']))
            # Keep strong references in entries to prevent Python object-ID reuse.
            # Writable arrays bypass the cache, including original feature frames.
            eligible = all(isinstance(a, np.ndarray) and a.flags.owndata
                and not a.flags.writeable for a in arrays)
            if not eligible:
                self.misses += 1
                return tracked_points(reference, current)
            points = tuple(k.pt for k in reference.keypoints)
            key = tuple(id(a) for a in arrays), points
            if key in self.entries:
                self.hits += 1
                self.entries.move_to_end(key)
                value = self.entries[key][2]
            else:
                self.misses += 1
                value = tracked_points(reference, current)
                self.entries[key] = (clocks[id(reference.gray)], arrays, deepcopy(value))
                while len(self.entries) > self.maximum_entries:
                    self.entries.popitem(last=False)
            # Callers can change their results without corrupting a future hit.
            return deepcopy(value)

        return bind(chained_points, tracked_points=cached_link)(frames)
