"""Batch projection arithmetic while retaining earliest complete patch witnesses.

Only small pose/query arrays are stacked. Historical pixel-prefix images stay
in their original storage and are queried in the original observation order.
"""
from copy import deepcopy
import numpy as np
from lewm.retained_floor_patch_development import RetainedFloorPatches, T, FOCAL

FRAME_BATCH = 32


def projected_rectangles(frames, xy, radius):
    corners = xy[:, None, :]+radius*np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    world = np.empty((len(frames), *corners.shape[:-1], 3), dtype=float)
    world[..., :2] = corners
    world[..., 2] = np.asarray([frame['floor_height'] for frame in frames])[:, None, None]
    positions = np.stack([frame['p'] for frame in frames])
    rotations = np.stack([frame['R'] for frame in frames])
    points = (world-positions[:, None, None, :])@rotations[:, None, :, :]
    camera = (points-T[:3, 3])@T[:3, :3]
    z = camera[..., 2]
    uv = camera[..., :2]/np.maximum(z[..., None], 1e-12)*FOCAL+[319.5, 239.5]
    lo = uv.min(2); hi = uv.max(2)
    margin = 1e-9+64*np.finfo(float).eps*np.maximum(np.abs(lo), np.abs(hi))
    lo -= margin; hi += margin
    visible = ((z >= .2)&(z <= 5.)).all(2)&(lo >= 0).all(2)&(hi < [639, 479]).all(2)
    a = np.floor(np.clip(lo, -1, 640)).astype(int)
    b = np.floor(np.clip(hi, -1, 640)).astype(int)
    return a, b, visible


def record_coverage(found, ids, frame, a, b, visible, radius):
    prefix = frame['prefix']
    for j in np.flatnonzero(visible):
        if found[int(ids[j])] is not None: continue
        x0, y0 = a[j]; x1, y1 = b[j]+1
        bad = int(prefix[y1, x1]-prefix[y0, x1]-prefix[y1, x0]+prefix[y0, x0])
        if bad == 0:
            found[int(ids[j])] = dict(witness=deepcopy(frame['witness']),
                pixel_rectangle=[a[j].tolist(), b[j].tolist()],
                nominal_enclosing_square_side_m=2*radius,
                all_projected_pixel_quads_measured_floor=True)


class BatchedRetainedFloorPatches(RetainedFloorPatches):
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
            for k, frame in enumerate(frames):
                record_coverage(found, ids, frame, a[k], b[k], visible[k], radius)
                if all(found[int(i)] is not None for i in ids): break
        return [dict(complete_nominal_foot_patch=value is not None, coverage_witness=value,
            retained_frames=len(self.frames), no_unobserved_pixel_inference=True,
            pose_or_terrain_uncertainty_certified=False, ground_support_approved=False) for value in found]
