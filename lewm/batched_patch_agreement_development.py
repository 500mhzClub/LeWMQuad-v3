"""Batch photometric arithmetic while retaining the original OpenCV samples."""
import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.direct_corner_flow_association_development import FLOW_RULES, patch_agrees
from lewm.rgbd_correspondence_motion_development import RULES, lift


def patches_agree(left, right, p, q):
    p, q = np.asarray(p), np.asarray(q)
    if p.ndim != 2 or p.shape[1:] != (2,) or q.shape != p.shape:
        raise SensorContractError('paired patch coordinates required')
    width = FLOW_RULES['patch_width']; half = width//2
    inside = np.ones(len(p), bool)
    for pixels in (p, q):
        inside &= ((pixels[:, 0] >= half) & (pixels[:, 0] <= 639-half)
            & (pixels[:, 1] >= half) & (pixels[:, 1] <= 479-half))
    indices = np.flatnonzero(inside); result = np.zeros(len(p), bool)
    if not len(indices): return result
    # Sampling still uses getRectSubPix on the uint8 image: preserve its rounding
    # before converting each sampled patch to the original float64 arithmetic.
    a = np.stack([cv2.getRectSubPix(left, (width, width), tuple(map(float, p[i])))
        for i in indices]).astype(float)
    b = np.stack([cv2.getRectSubPix(right, (width, width), tuple(map(float, q[i])))
        for i in indices]).astype(float)
    a -= a.mean(axis=(1, 2), keepdims=True); b -= b.mean(axis=(1, 2), keepdims=True)
    spread = np.minimum(a.std(axis=(1, 2)), b.std(axis=(1, 2)))
    denominator = np.sqrt(np.sum(a*a, axis=(1, 2))*np.sum(b*b, axis=(1, 2)))
    correlation = np.divide(np.sum(a*b, axis=(1, 2)), denominator,
        out=np.zeros(len(indices)), where=denominator>0)
    minimum_std = FLOW_RULES['minimum_patch_std']; minimum_zncc = FLOW_RULES['minimum_patch_zncc']
    result[indices] = (spread >= minimum_std) & (correlation >= minimum_zncc)
    # Scalar fallback at decision boundaries avoids changing a threshold decision
    # through a different floating-point reduction order.
    near = (np.abs(spread-minimum_std) <= 1e-10) | (np.abs(correlation-minimum_zncc) <= 1e-10)
    for i in indices[near]: result[i] = patch_agrees(left, right, p[i], q[i])
    return result


def tracked_points(reference, current):
    """Original direct association with only its photometric loop batched."""
    for f in (reference, current):
        if not isinstance(f.gray, np.ndarray) or f.gray.shape != (480, 640) or f.gray.dtype != np.uint8:
            raise SensorContractError('exact finite uint8 camera grayscale required')
    if len(reference.keypoints) > FLOW_RULES['maximum_reference_corners']:
        raise SensorContractError('bounded original reference-corner population required')
    points = []; seen = set()
    for keypoint in reference.keypoints:
        p = np.asarray(keypoint.pt, float)
        if p.shape != (2,) or not np.isfinite(p).all(): raise SensorContractError('finite reference corners required')
        location = tuple(np.rint(p*2).astype(int))
        if location not in seen: points.append(p); seen.add(location)
    counts = dict(reference_corners=len(reference.keypoints), distinct_reference_corners=len(points),
        forward_finite=0, reverse_finite=0, forward_backward=0, photometric=0, distinct_current=0, valid_depth_pair=0)
    receipt = dict(counts=counts, association='direct_corner_lk_photometric_v1',
        original_descriptor_gates_applied=False, initial_flow='same_pixel_hypothesis',
        flow_rules=dict(FLOW_RULES), forward_backward_limit_pixels=RULES['fb_pixels'],
        rigid_geometry_evaluated=False, pose_admitted=False, native_state_used=False)
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 2)))
    if not points: return empty, receipt
    p = np.asarray(points, np.float32).reshape(-1, 1, 2)
    options = dict(winSize=(RULES['lk_window'],)*2, maxLevel=RULES['lk_levels'],
        criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS, 30, .01))
    q, ok, _ = cv2.calcOpticalFlowPyrLK(reference.gray, current.gray, p, None, **options)
    if q is None or ok is None or q.shape != p.shape or ok.shape != (len(p), 1):
        raise SensorContractError('complete forward flow response required')
    good = ok.ravel().astype(bool)&np.isfinite(q).all((1, 2)); counts['forward_finite'] = int(good.sum())
    p = p[good]; q = q[good]
    if not len(p): return empty, receipt
    back, ok, _ = cv2.calcOpticalFlowPyrLK(current.gray, reference.gray, q, None, **options)
    if back is None or ok is None or back.shape != p.shape or ok.shape != (len(p), 1):
        raise SensorContractError('complete reverse flow response required')
    good = ok.ravel().astype(bool)&np.isfinite(back).all((1, 2)); counts['reverse_finite'] = int(good.sum())
    good &= np.linalg.norm(back[:, 0]-p[:, 0], axis=1) <= RULES['fb_pixels']
    counts['forward_backward'] = int(good.sum()); p = p[good, 0]; q = q[good, 0]
    good = patches_agree(reference.gray, current.gray, p, q)
    counts['photometric'] = int(good.sum()); p = p[good]; q = q[good]
    keep = []; seen = set()
    for i, point in enumerate(q):
        location = tuple(np.rint(point*2).astype(int))
        if location not in seen: keep.append(i); seen.add(location)
    p = p[keep]; q = q[keep]; counts['distinct_current'] = len(p)
    a, va = lift(reference.depth, p); b, vb = lift(current.depth, q)
    valid = va & vb & np.isfinite(a).all(1) & np.isfinite(b).all(1)
    counts['valid_depth_pair'] = int(valid.sum())
    return (a[valid], b[valid], p[valid], q[valid]), receipt
