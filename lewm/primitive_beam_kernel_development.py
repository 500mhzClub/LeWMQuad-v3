"""CPU pixel-beam/box reduction; no fast-math, parallelism or persistent cache."""
import numpy as np
from numba import njit

from lewm.causal_depth_observation_development import FOCAL


@njit(cache=False, nogil=True, boundscheck=True)
def reduce_primitive_beams(depth, valid, floor, optical_low, optical_high, rect_low, rect_high,
                           complete, range_error, minimum_valid_depth):
    count = len(optical_low)
    clear = np.zeros(count, np.bool_); conflict = clear.copy()
    scanned = np.zeros(count, np.int64); floor_pixels = scanned.copy()
    for i in range(count):
        low, high = optical_low[i], optical_high[i]
        x0, y0 = rect_low[i, 0], rect_low[i, 1]
        x1, y1 = rect_high[i, 0], rect_high[i, 1]
        if low[2] <= 0 <= high[2]: x0, y0, x1, y1 = 0, 0, 639, 479
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(639, x1), min(479, y1)
        if x0 > x1 or y0 > y1 or high[2] + range_error < .2: continue
        if not complete[i] and high[2] + range_error < minimum_valid_depth: continue
        can_clear = complete[i]
        for y in range(y0, y1+1):
            for x in range(x0, x1+1):
                entry, exit = max(0., low[2]), high[2]
                feasible = True
                for axis in range(4):
                    if axis == 0: coefficient, limit = (x - 319.5) / FOCAL, high[0]
                    elif axis == 1: coefficient, limit = -(x + 1 - 319.5) / FOCAL, -low[0]
                    elif axis == 2: coefficient, limit = (y - 239.5) / FOCAL, high[1]
                    else: coefficient, limit = -(y + 1 - 239.5) / FOCAL, -low[1]
                    if coefficient > 0: exit = min(exit, limit / coefficient)
                    elif coefficient < 0: entry = max(entry, limit / coefficient)
                    elif limit < 0: feasible = False
                margin = 1e-10 + 128 * np.finfo(np.float64).eps * max(abs(entry), abs(exit))
                entry -= margin; exit += margin
                if not feasible or entry > exit or exit <= 0: continue
                scanned[i] += 1
                is_floor = floor[y, x]
                if is_floor: floor_pixels[i] += 1
                minimum, maximum = np.inf, -np.inf
                any_valid, all_valid = False, True
                for corner in range(4):
                    if corner == 0: yy, xx = y, x
                    elif corner == 1: yy, xx = y, x+1
                    elif corner == 2: yy, xx = y+1, x+1
                    else: yy, xx = y+1, x
                    if yy >= 480 or xx >= 640 or not valid[yy, xx]:
                        all_valid = False
                    else:
                        any_valid = True
                        value = np.float64(depth[yy, xx])
                        minimum = min(minimum, value); maximum = max(maximum, value)
                beyond = minimum - range_error > exit
                possible_near = maximum + range_error >= entry and minimum - range_error <= exit
                if any_valid and not is_floor and possible_near: conflict[i] = True
                if not all_valid or not (is_floor or beyond): can_clear = False
        clear[i] = can_clear and scanned[i] > 0 and not conflict[i]
    return clear, conflict, scanned, floor_pixels


def warm_primitive_beam_kernel():
    """Compile with runtime array layouts before collecting an observation."""
    depth = np.zeros((480, 640), np.float32)
    valid = np.zeros((480, 640), np.bool_); floor = valid.copy()
    for array in (depth, valid, floor): array.flags.writeable = False
    low = np.empty((0, 3), np.float64); rect = np.empty((0, 2), np.int64)
    reduce_primitive_beams(depth, valid, floor, low, low, rect, rect,
                           np.empty(0, np.bool_), 0., np.inf)
