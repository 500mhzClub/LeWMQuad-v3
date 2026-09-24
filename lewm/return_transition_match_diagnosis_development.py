"""Read-only stage counts for the unchanged corner/SIFT/LK correspondence path."""
import cv2
import numpy as np
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.rgbd_correspondence_motion_development import RULES, lift, cells


def diagnose_pair(reference, current):
    counts = dict(reference_features=len(reference.keypoints), current_features=len(current.keypoints),
        forward_ratio=0, backward_ratio=0, mutual_unique=0, flow_status_finite=0,
        forward_backward=0, descriptor_location_consistent=0, depth_lifted=0,
        reference_grid_cells=0, current_grid_cells=0)
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 2)))
    measured = empty
    d0, d1 = reference.descriptors, current.descriptors
    if d0 is not None and d1 is not None and min(len(d0), len(d1)) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        def ratios(a, b):
            return {x.queryIdx: x.trainIdx for pair in matcher.knnMatch(a, b, k=2) if len(pair) == 2
                    for x, y in [pair] if x.distance < RULES['ratio'] * y.distance}
        forward, backward = ratios(d0, d1), ratios(d1, d0)
        counts.update(forward_ratio=len(forward), backward_ratio=len(backward))
        pairs, seen0, seen1 = [], set(), set()
        for i, j in sorted(forward.items()):
            if backward.get(j) != i:
                continue
            left = tuple(np.rint(np.asarray(reference.keypoints[i].pt) * 2).astype(int))
            right = tuple(np.rint(np.asarray(current.keypoints[j].pt) * 2).astype(int))
            if left in seen0 or right in seen1:
                continue
            pairs.append((i, j)); seen0.add(left); seen1.add(right)
        counts['mutual_unique'] = len(pairs)
        if pairs:
            p = np.float32([reference.keypoints[i].pt for i, j in pairs]).reshape(-1, 1, 2)
            initial = np.float32([current.keypoints[j].pt for i, j in pairs]).reshape(-1, 1, 2)
            options = dict(winSize=(RULES['lk_window'],) * 2, maxLevel=RULES['lk_levels'],
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            q, ok, _ = cv2.calcOpticalFlowPyrLK(reference.gray, current.gray, p, initial.copy(), **options)
            back, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(current.gray, reference.gray, q, p.copy(), **options)
            good = ok.ravel().astype(bool) & reverse_ok.ravel().astype(bool)
            good &= np.isfinite(q).all((1, 2)) & np.isfinite(back).all((1, 2))
            counts['flow_status_finite'] = int(good.sum())
            good &= np.linalg.norm(back[:, 0] - p[:, 0], axis=1) <= RULES['fb_pixels']
            counts['forward_backward'] = int(good.sum())
            good &= np.linalg.norm(q[:, 0] - initial[:, 0], axis=1) <= RULES['reprojection_pixels']
            counts['descriptor_location_consistent'] = int(good.sum())
            ua, ub = p[good, 0], q[good, 0]
            a, va = lift(reference.depth, ua); b, vb = lift(current.depth, ub); use = va & vb
            measured = a[use], b[use], ua[use], ub[use]
    original = matched_points(reference, current)
    if not all(np.array_equal(a, b) for a, b in zip(measured, original)):
        raise ValueError('diagnostic correspondence arrays differ from frozen implementation')
    counts.update(depth_lifted=len(measured[0]), reference_grid_cells=cells(measured[2]),
        current_grid_cells=cells(measured[3]), exact_original_arrays=True,
        minimum_match_count_pass=len(measured[0]) >= RULES['minimum_matches'], pose_acceptance_claim=False)
    return counts
