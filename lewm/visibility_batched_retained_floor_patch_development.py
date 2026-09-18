"""Skip empty per-frame visibility enumeration without changing coverage rules."""
import numpy as np
from lewm.batched_retained_floor_patch_development import (
    BatchedRetainedFloorPatches, projected_rectangles, record_coverage, FRAME_BATCH)


class VisibilityBatchedRetainedFloorPatches(BatchedRetainedFloorPatches):
    def coverage(self, centres, radius=.022):
        xy = np.asarray(centres, float)
        if (xy.ndim != 2 or xy.shape[1:] != (2,) or len(xy) > 128 or not np.isfinite(xy).all()
                or (xy.size and np.max(np.abs(xy)) > 4.9) or not np.isfinite(radius) or not 0 < radius <= .1):
            raise ValueError('bounded nominal foot centres and radius required')
        found = [None]*len(xy)
        for start in range(0, len(self.frames), FRAME_BATCH):
            ids = np.asarray([i for i, value in enumerate(found) if value is None], int)
            if not len(ids): break
            frames = self.frames[start:start+FRAME_BATCH]
            try:
                # An unused later frame must not introduce an arithmetic error
                # that chronological early termination would have avoided.
                with np.errstate(all='raise'):
                    a, b, visible = projected_rectangles(frames, xy[ids], radius)
            except (FloatingPointError, ValueError, TypeError, KeyError, IndexError, OverflowError):
                for frame in frames:
                    remaining = np.asarray([i for i in ids if found[int(i)] is None], int)
                    if not len(remaining): break
                    one_a, one_b, one_visible = projected_rectangles([frame], xy[remaining], radius)
                    record_coverage(found, remaining, frame, one_a[0], one_b[0], one_visible[0], radius)
                continue
            any_visible = visible.any(axis=1)
            for k, frame in enumerate(frames):
                if any_visible[k]:
                    record_coverage(found, ids, frame, a[k], b[k], visible[k], radius)
                else:
                    # record_coverage always accesses the prefix, even when
                    # no pixel is visible. Preserve that failure behavior.
                    frame['prefix']
                if all(found[int(i)] is not None for i in ids): break
        return [dict(complete_nominal_foot_patch=value is not None, coverage_witness=value,
            retained_frames=len(self.frames), no_unobserved_pixel_inference=True,
            pose_or_terrain_uncertainty_certified=False, ground_support_approved=False) for value in found]
